use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};
use sztorm::agent::{AgentIdentifier};
use sztorm::error::{InternalGameError, SztormError};
use sztorm::domain::{Action, DomainParameters};
use sztorm::state::StateUpdate;
use crate::prisoner::domain::PrisonerId::{Andrzej, Janusz};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrisonerAction{
    Betray,
    Cover
}

impl Display for PrisonerAction {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if f.alternate(){
            match self{
                PrisonerAction::Betray => write!(f, "B"),
                PrisonerAction::Cover => write!(f, "C")
            }
        } else{
            write!(f, "{:?}", self)
        }

    }
}


impl Action for PrisonerAction{}
//--------------------------------------


#[derive(thiserror::Error, Debug, PartialEq, Clone)]
pub enum PrisonerError{
    #[error("Performed different action (chosen: {chosen:?}, logged: {logged:?})")]
    DifferentActionPerformed{
        chosen: PrisonerAction,
        logged: PrisonerAction
    },
    #[error("Environment logged action {0}, but none was performed")]
    NoLastAction(PrisonerAction),
    #[error("Player: {0} played after GameOver")]
    ActionAfterGameOver(PrisonerId),
    #[error("Player: {0} played out of order")]
    ActionOutOfOrder(PrisonerId),
}


impl Into<SztormError<PrisonerDomain>> for PrisonerError {
    fn into(self) -> SztormError<PrisonerDomain> {
        SztormError::Game(self)
    }
}



impl InternalGameError<PrisonerDomain> for PrisonerError{

}


#[derive(Clone, Debug)]
pub struct PrisonerDomain;
#[derive(Debug, Copy, Clone)]
pub struct PrisonerUpdate{
    pub own_action: PrisonerAction,
    pub other_prisoner_action: PrisonerAction}

impl Display for PrisonerUpdate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Update [own action: {}, opponent's action: {}]", self.own_action, self.other_prisoner_action)
    }
}

impl StateUpdate for PrisonerUpdate{}

//pub type PrisonerId = u8;
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PrisonerId{
    Andrzej,
    Janusz
}

impl PrisonerId{
    pub fn other(self) -> Self{
        match self{
            Self::Janusz => Andrzej,
            Self::Andrzej => Janusz
        }
    }
}



impl AgentIdentifier for PrisonerId{}

#[derive(Debug, Copy, Clone, Default)]
pub struct PrisonerMap<T>{
    andrzej_s: T,
    janusz_s: T
}
impl<T> Display for PrisonerMap<T> where T: Display{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Andrzej: {} | Janusz:{}]", self[Andrzej], self[Janusz])
    }
}

impl Display for PrisonerId{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T> PrisonerMap<T>{
    pub fn new(andrzej_s: T, janusz_s: T) -> Self{
        Self{andrzej_s, janusz_s}
    }

}

impl<T> Index<PrisonerId> for PrisonerMap<T>{
    type Output = T;

    fn index(&self, index: PrisonerId) -> &Self::Output {
        match index{
            PrisonerId::Andrzej => &self.andrzej_s,
            PrisonerId::Janusz => &self.janusz_s
        }
    }
}

impl<T> IndexMut<PrisonerId> for PrisonerMap<T>{

    fn index_mut(&mut self, index: PrisonerId) -> &mut Self::Output {
        match index{
            PrisonerId::Andrzej => &mut self.andrzej_s,
            PrisonerId::Janusz => &mut self.janusz_s
        }
    }
}


pub const PRISONERS:[PrisonerId;2] = [PrisonerId::Andrzej, PrisonerId::Janusz];

pub type PrisonerReward = i32;


impl DomainParameters for PrisonerDomain{
    type ActionType = PrisonerAction;
    type GameErrorType = PrisonerError;
    type UpdateType = PrisonerUpdate;
    type AgentId = PrisonerId;
    type UniversalReward = PrisonerReward;
}
