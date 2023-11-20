use std::rc::Rc;
use std::sync::Arc;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use amfi::domain::DomainParameters;
use amfi::env::{EnvironmentStateUniScore, EnvStateSequential};
use crate::classic::common::{AsymmetricRewardTable, AsymmetricRewardTableInt, Side};
use crate::classic::domain::{ClassicAction, ClassicGameDomain, ClassicGameDomainNumbered, ClassicGameError, EncounterReport, EncounterReportNamed, EncounterReportNumbered, IntReward};
use crate::classic::domain::ClassicGameError::ActionAfterGameOver;
use log::{debug};
use std::fmt::{Display, Formatter};

pub type AgentNum = u32;
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
pub struct PlayerPairing {
    pub paired_player: AgentNum,
    pub taken_action: Option<ClassicAction>,
    pub side: Side
}

impl Display for PlayerPairing{
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

pub type PairingVec = Vec<PlayerPairing>;

#[derive(Debug, Clone)]
pub struct PairingState{
    actual_pairings: PairingVec,
    previous_pairings: Vec<Arc<PairingVec>>,
    target_rounds: usize,
    indexes: Vec<u32>,
    reward_table: AsymmetricRewardTableInt,
    score_cache: Vec<i32>,
    current_player_index: usize,


}

impl PairingState{
    pub fn new_even(players: u32, target_rounds: usize, reward_table: AsymmetricRewardTableInt) -> Result<Self, ClassicGameError<AgentNum>>{
        /*
        if players & 0x01 != 0{
            return Err(ClassicGameError::ExpectedEvenNumberOfPlayers(players));
        }


         */

        let mut indexes: Vec<u32> = (0..players as u32).into_iter().collect();
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
        })
    }

    fn create_pairings(indexes: &[u32]) -> Result<PairingVec, ClassicGameError<AgentNum>>{
        if indexes.len() & 0x01 != 0{
            return Err(ClassicGameError::ExpectedEvenNumberOfPlayers(indexes.len() as u32));
        } else {
            let mut v = Vec::with_capacity(indexes.len());
            v.resize_with(indexes.len(), || PlayerPairing::default());
            for i in 0..indexes.len(){
                let index:usize = indexes[i] as usize;
                if i & 0x01 == 0{

                    
                    //even
                    v[index] = PlayerPairing{
                        paired_player: indexes[i+1],
                        taken_action: None,
                        side: Side::Left,
                    }

                } else {
                    
                    v[index] = PlayerPairing{
                        paired_player: indexes[i-1],
                        taken_action: None,
                        side: Side::Right,
                    }
                }
            }
            Ok(v)
        }

    }

    fn prepare_new_pairing(&mut self) -> Result<(), ClassicGameError<AgentNum>>{

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

impl EnvStateSequential<ClassicGameDomainNumbered> for PairingState {
    type Updates = Vec<(AgentNum, Arc<Vec<EncounterReportNumbered>>)>;

    fn current_player(&self) -> Option<AgentNum> {
        if self.current_player_index  < self.actual_pairings.len(){
            Some(self.current_player_index as u32)
        } else {
            None
        }
    }

    fn is_finished(&self) -> bool {
        self.previous_pairings.len() >= self.target_rounds
    }

    fn forward(&mut self, agent: AgentNum, action: ClassicAction)
        -> Result<Self::Updates, ClassicGameError<AgentNum>> {
        if let Some(destined_agent) = self.current_player(){
            if destined_agent == agent{
                self.actual_pairings[agent as usize].taken_action = Some(action);
                let this_pairing = self.actual_pairings[agent as usize];
                let other_player_index = this_pairing.paired_player;
                let other_pairing = self.actual_pairings[other_player_index as usize];
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
                    self.score_cache[agent as usize] += rewards_reoriented.0;
                    self.score_cache[other_player_index as usize] += rewards_reoriented.1;

                }
                //set next index
                self.current_player_index +=1;

                if self.current_player_index >= self.actual_pairings.len(){


                    let encounters_vec: Vec<EncounterReportNumbered> = (0..self.actual_pairings.len())
                        .into_iter().map(|i|{
                        let actual_pairing = self.actual_pairings[i];
                        let other_player = self.actual_pairings[i].paired_player;
                        let reverse_pairing = self.actual_pairings[other_player as usize];
                        EncounterReport{
                            own_action: self.actual_pairings[i].taken_action.unwrap(),
                            other_player_action: self.actual_pairings[other_player as usize].taken_action.unwrap(),
                            side: actual_pairing.side,
                            other_id: other_player,
                        }
                    }).collect();
                    let encounters = Arc::new(encounters_vec);

                    self.prepare_new_pairing()?;
                    self.current_player_index = 0;

                    let updates: Vec<(AgentNum, Arc<Vec<EncounterReportNumbered>>)> = (0..self.actual_pairings.len())
                        .into_iter().map(|i|{
                        (i as u32, encounters.clone())
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

impl EnvironmentStateUniScore<ClassicGameDomainNumbered> for PairingState{
    fn state_score_of_player(&self, agent: &AgentNum) -> IntReward {
        self.score_cache[*agent as usize]
    }
}