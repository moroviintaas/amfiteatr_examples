use std::rc::Rc;
use std::sync::Arc;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use amfi::domain::DomainParameters;
use amfi::env::{EnvironmentStateUniScore, EnvStateSequential};
use crate::classic::common::{AsymmetricRewardTable, AsymmetricRewardTableInt};
use crate::classic::domain::{ClassicAction, ClassicGameDomain, ClassicGameDomainNumbers, ClassicGameError, EncounterUpdate, IntReward};

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
    pub pair: Option<AgentNum>,
    pub taken_action: Option<ClassicAction>
}

pub type PairingVec = Vec<PlayerPairing>;

#[derive(Debug, Clone)]
pub struct PairingState{
    actual_pairings: PairingVec,
    actual_resolved: usize,
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
        let actual_pairings = Self::create_pairings(&indexes[..])?;

        let mut score_cache = Vec::with_capacity(indexes.len());
        score_cache.resize_with(indexes.len(), || 0);
        Ok(Self{
            actual_pairings,
            indexes,
            target_rounds,
            actual_resolved: 0,
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
                if i & 0x01 == 0{
                    //even
                    v[i] = PlayerPairing{
                        pair: Some(indexes[i+1]),
                        taken_action: None,
                    }

                } else {
                    v[i] = PlayerPairing{
                        pair: Some(indexes[i-1]),
                        taken_action: None
                    }
                }
            }
            Ok(v)
        }

    }

    fn prepare_new_pairing(&mut self) -> Result<(), ClassicGameError<AgentNum>>{

        let mut rng = thread_rng();
        self.indexes.shuffle(&mut rng);
        let mut pairings = Self::create_pairings(&self.indexes[..])?;
        std::mem::swap(&mut pairings, &mut self.actual_pairings);
        self.previous_pairings.push(Arc::new(pairings));
        Ok(())

    }

    pub fn is_round_clean(&self) -> bool{
        self.actual_resolved == 0
    }

}

impl EnvStateSequential<ClassicGameDomainNumbers> for PairingState {
    type Updates = Vec<(AgentNum, EncounterUpdate)>;

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
        todo!()

    }
}

impl EnvironmentStateUniScore<ClassicGameDomainNumbers> for PairingState{
    fn state_score_of_player(&self, agent: &AgentNum) -> IntReward {
        self.score_cache[*agent as usize]
    }
}