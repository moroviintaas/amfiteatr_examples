use amfi::domain::DomainParameters;
use crate::classic::domain::{ClassicAction, ClassicGameError};

pub type AgentNum = u32;
#[derive(Debug, Clone)]
pub struct PairDomain;


impl DomainParameters for PairDomain{
    type ActionType = ClassicAction;
    type GameErrorType = ClassicGameError;
    type UpdateType = Vec<(AgentNum, ClassicAction)>;
    type AgentId = AgentNum;
    type UniversalReward = i32;
}



#[derive(Copy, Clone, Debug)]
pub struct PlayerPairing {
    pub pair: Option<AgentNum>,
    pub taken_action: Option<ClassicAction>
}

#[derive(Debug, Clone)]
pub struct PairingState{
    actual_pairings: Vec<PlayerPairing>,
    actual_resolved: usize,
    previous_pairings: Vec<Vec<PlayerPairing>>,
    target_rounds: usize,


}
/*
impl PairingState{
    pub fn new(players: usize, target_rounds: usize) -> Self{

    }

}*/