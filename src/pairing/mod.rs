use std::rc::Rc;
use std::sync::Arc;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use amfi::domain::{DomainParameters, Renew};
use amfi::env::{EnvironmentStateUniScore, EnvStateSequential};
use crate::classic::common::{AsymmetricRewardTable, AsymmetricRewardTableInt, Side};
use crate::classic::domain::{AsUsize, ClassicAction, ClassicGameDomain, ClassicGameDomainNumbered, ClassicGameError, ClassicGameUpdate, EncounterReport, EncounterReportNamed, EncounterReportNumbered, IntReward, UsizeAgentId};
use crate::classic::domain::ClassicGameError::ActionAfterGameOver;
use log::{debug};
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;

pub type AgentNum = u32;

impl AsUsize for AgentNum{
    fn as_usize(&self) -> usize {
        *self as usize
    }

    fn make_from_usize(u: usize) -> Self {
        u as AgentNum
    }
}

#[derive(Debug, Clone)]
pub struct PairDomain;

/*
impl DomainParameters for PairDomain{
    type ActionType = ClassicAction;
    type GameErrorType = ClassicGameError<AgentNum>;
    type UpdateType = Vec<(AgentNum, ClassicAction)>;
    type AgentId = AgentNum;
    type UniversalReward = i32;
}

 */



#[derive(Copy, Clone, Debug, Default)]
pub struct PlayerPairing<ID: UsizeAgentId> {
    pub paired_player: ID,
    pub taken_action: Option<ClassicAction>,
    pub side: Side
}

impl<ID: UsizeAgentId> Display for PlayerPairing<ID>{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        //write!(f, "({}-{})", self.id)
        let a = match self.taken_action{
            None => String::from("0"),
            Some(a) => format!("{:?}", a)
        };
        match self.side{
            Side::Left => write!(f, "[{} -> {}]", &a, self.paired_player),
            Side::Right => write!(f, "[{} <- {}]", self.paired_player, &a),

        }
    }
}

pub type PairingVec<ID> = Vec<PlayerPairing<ID>>;

#[derive(Debug, Clone)]
pub struct PairingState<ID: UsizeAgentId>{
    actual_pairings: PairingVec<ID>,
    previous_pairings: Vec<Arc<PairingVec<ID>>>,
    target_rounds: usize,
    indexes: Vec<usize>,
    reward_table: AsymmetricRewardTableInt,
    score_cache: Vec<i32>,
    current_player_index: usize,
    _id: PhantomData<ID>


}

pub type PairingStateNumbered = PairingState<AgentNum>;

impl<ID: UsizeAgentId> PairingState<ID>{
    pub fn new_even(players: usize, target_rounds: usize, reward_table: AsymmetricRewardTableInt) -> Result<Self, ClassicGameError<ID>>{
        /*
        if players & 0x01 != 0{
            return Err(ClassicGameError::ExpectedEvenNumberOfPlayers(players));
        }


         */

        let mut indexes: Vec<usize> = (0..players).into_iter().collect();
        let mut rng = thread_rng();
        indexes.shuffle(&mut rng);
        //debug!("Shuffled indexes: {:?}", &indexes);
        //println!("Shuffled indexes: {:?}", &indexes);
        let actual_pairings = Self::create_pairings(&indexes[..])?;

        let mut score_cache = Vec::with_capacity(indexes.len());
        score_cache.resize_with(indexes.len(), || 0);
        Ok(Self{
            actual_pairings,
            indexes,
            target_rounds,
            previous_pairings: Vec::with_capacity(target_rounds),
            reward_table,
            score_cache,
            current_player_index: 0,
            _id: PhantomData::default()
        })
    }

    fn create_pairings(indexes: &[usize]) -> Result<PairingVec<ID>, ClassicGameError<ID>>{
        if indexes.len() & 0x01 != 0{
            return Err(ClassicGameError::ExpectedEvenNumberOfPlayers(indexes.len() as u32));
        } else {
            let mut v = Vec::with_capacity(indexes.len());
            v.resize_with(indexes.len(), || PlayerPairing{
                paired_player: ID::make_from_usize(0),
                taken_action: None,
                side: Default::default(),
            }) ;
            for i in 0..indexes.len(){
                let index:usize = indexes[i] as usize;
                if i & 0x01 == 0{

                    
                    //even
                    v[index] = PlayerPairing{
                        paired_player: ID::make_from_usize(indexes[i+1] as usize),
                        taken_action: None,
                        side: Side::Left,
                    }

                } else {
                    
                    v[index] = PlayerPairing{
                        paired_player: ID::make_from_usize(indexes[i-1] as usize),
                        taken_action: None,
                        side: Side::Right,
                    }
                }
            }
            Ok(v)
        }

    }

    fn prepare_new_pairing(&mut self) -> Result<(), ClassicGameError<ID>>{

        let mut rng = thread_rng();
        self.indexes.shuffle(&mut rng);
        //debug!("Shuffled indexes: {:?}", &self.indexes);
        //println!("Shuffled indexes: {:?}", &self.indexes);
        let mut pairings = Self::create_pairings(&self.indexes[..])?;
        std::mem::swap(&mut pairings, &mut self.actual_pairings);
        //debug!("Pairings: {:?}", &self.actual_pairings);
        //println!("Pairings: {:?}", &self.actual_pairings);
        self.previous_pairings.push(Arc::new(pairings));

        
        Ok(())

    }

    pub fn is_round_clean(&self) -> bool{
        self.current_player_index == 0
    }

}

impl<ID: UsizeAgentId> Display for PairingState<ID>{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        /*write!(f, "Rounds: {} |", self.previous_pairings.len())?;
        let mut s = self.previous_pairings.iter().fold(String::new(), |mut acc, update| {
            acc.push_str(&format!("({}){:#}-{:#}\n", update.side, update.own_action, update.other_player_action));
            acc
        });
        s.pop();
        write!(f, "{}", s)*/

        for r in 0..self.previous_pairings.len(){
            write!(f, "Round: {r:}:\n")?;
            for i in 0..self.previous_pairings[r].len(){
                write!(f, "\t{}\tpositioned: {:?}\tpaired with: {}\t;",
                       i, self.previous_pairings[r][i].side, self.previous_pairings[r][i].paired_player)?;
                if let Some(action) = self.previous_pairings[r][i].taken_action{
                    write!(f, "taken action: {action:?}\t")?;
                }
                else{
                    write!(f, "taken action: ---\t")?;
                }
                let other_index = self.previous_pairings[r][i].paired_player.as_usize();
                if let Some(action) = self.previous_pairings[r][other_index].taken_action{
                    write!(f, "against: {action:?}\t")?;
                }
                else{
                    write!(f, "against: ---\t")?;
                }
                write!(f, "\n")?;
            }
        }
        write!(f, "")
    }
}

impl<ID: UsizeAgentId> EnvStateSequential<ClassicGameDomain<ID>> for PairingState<ID> {
    type Updates = Vec<(ID, ClassicGameUpdate<ID>)>;

    fn current_player(&self) -> Option<ID> {
        if self.current_player_index  < self.actual_pairings.len(){
            Some(ID::make_from_usize(self.current_player_index))
        } else {
            None
        }
    }

    fn is_finished(&self) -> bool {
        self.previous_pairings.len() >= self.target_rounds
    }

    fn forward(&mut self, agent: ID, action: ClassicAction)
        -> Result<Self::Updates, ClassicGameError<ID>> {
        if let Some(destined_agent) = self.current_player(){
            if destined_agent == agent{
                self.actual_pairings[agent.as_usize()].taken_action = Some(action);
                let this_pairing = self.actual_pairings[agent.as_usize()];
                let other_player_index = this_pairing.paired_player;
                let other_pairing = self.actual_pairings[other_player_index.as_usize()];
                // possibly update score cache if other player played already
                if let Some(other_action) = other_pairing.taken_action {
                    let (left_action, right_action) = match this_pairing.side{
                        Side::Left => (action, other_action),
                        Side::Right => (other_action, action)
                    };
                    let rewards = self.reward_table.rewards(left_action, right_action);
                    let rewards_reoriented = match this_pairing.side{
                        Side::Left => rewards,
                        Side::Right => (rewards.1, rewards.0)
                    };
                    self.score_cache[agent.as_usize()] += rewards_reoriented.0;
                    self.score_cache[other_player_index.as_usize()] += rewards_reoriented.1;

                }
                //set next index
                self.current_player_index +=1;

                if self.current_player_index >= self.actual_pairings.len(){


                    let encounters_vec: Vec<EncounterReport<ID>> = (0..self.actual_pairings.len())
                        .into_iter().map(|i|{
                        let actual_pairing = self.actual_pairings[i];
                        let other_player = self.actual_pairings[i].paired_player;
                        let reverse_pairing = self.actual_pairings[other_player.as_usize()];
                        EncounterReport{
                            own_action: self.actual_pairings[i].taken_action.unwrap(),
                            other_player_action: self.actual_pairings[other_player.as_usize()].taken_action.unwrap(),
                            side: actual_pairing.side,
                            other_id: other_player,
                        }
                    }).collect();
                    let encounters = Arc::new(encounters_vec);

                    self.prepare_new_pairing()?;
                    self.current_player_index = 0;

                    let opairings = match self.is_finished(){
                        true => None,
                        false => Some(Arc::new(self.actual_pairings.clone()))
                    };
                    let singe_update = ClassicGameUpdate{
                        encounters,
                        pairing: opairings,
                    };
                    let updates: Vec<(ID, ClassicGameUpdate<ID>)> = (0..self.actual_pairings.len())
                        .into_iter().map(|i|{
                        (ID::make_from_usize(i), singe_update.clone())
                    }).collect();

                    Ok(updates)

                } else{
                    Ok(Vec::default())
                }





            } else{
                Err(ClassicGameError::ViolatedOrder(agent))
            }

        } else {
            Err(ActionAfterGameOver(agent))
        }

    }
}

impl<ID: UsizeAgentId> EnvironmentStateUniScore<ClassicGameDomain<ID>> for PairingState<ID>{
    fn state_score_of_player(&self, agent: &ID) -> IntReward {
        self.score_cache[agent.as_usize()]
    }
}

impl<ID: UsizeAgentId> Renew<()> for PairingState<ID>{
    fn renew_from(&mut self, _base: ()) {
        self.score_cache.clear();
        self.previous_pairings.clear();
        self.current_player_index = 0;
        self.prepare_new_pairing().unwrap();
    }
}