use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};
use amfi::agent::{AgentIdentifier};
use amfi::error::{AmfiError};
use amfi::domain::{Action, DomainParameters};
use crate::classic::domain::PrisonerId::{Andrzej, Janusz};
use enum_map::Enum;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Enum, Serialize, Deserialize)]
pub enum ClassicAction {
    Defect,
    Cooperate
}

impl Display for ClassicAction {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if f.alternate(){
            match self{
                ClassicAction::Defect => write!(f, "B"),
                ClassicAction::Cooperate => write!(f, "C")
            }
        } else{
            write!(f, "{:?}", self)
        }

    }
}


impl Action for ClassicAction {}
//--------------------------------------


#[derive(thiserror::Error, Debug, PartialEq, Clone)]
pub enum ClassicGameError {
    #[error("Performed different action (chosen: {chosen:?}, logged: {logged:?})")]
    DifferentActionPerformed{
        chosen: ClassicAction,
        logged: ClassicAction
    },
    #[error("Environment logged action {0}, but none was performed")]
    NoLastAction(ClassicAction),
    #[error("Player: {0} played after GameOver")]
    ActionAfterGameOver(PrisonerId),
    #[error("Player: {0} played out of order")]
    ActionOutOfOrder(PrisonerId),
    #[error("Value can't be probability: {0}")]
    NotAProbability(f64),
}

/*
impl Into<AmfiError<PrisonerDomain>> for PrisonerError {
    fn into(self) -> AmfiError<PrisonerDomain> {
        AmfiError::Game(self)
    }
}

 */
impl From<ClassicGameError> for AmfiError<ClassicGameDomain>{
    fn from(value: ClassicGameError) -> Self {
        AmfiError::Game(value)
    }
}


#[derive(Clone, Debug)]
pub struct ClassicGameDomain;
#[derive(Debug, Copy, Clone)]
pub struct PrisonerUpdate{
    pub own_action: ClassicAction,
    pub other_prisoner_action: ClassicAction
}

impl Display for PrisonerUpdate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Update [own action: {}, opponent's action: {}]", self.own_action, self.other_prisoner_action)
    }
}

//impl StateUpdate for PrisonerUpdate{}

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

pub type IntReward = i32;


impl DomainParameters for ClassicGameDomain {
    type ActionType = ClassicAction;
    type GameErrorType = ClassicGameError;
    type UpdateType = PrisonerUpdate;
    type AgentId = PrisonerId;
    type UniversalReward = IntReward;
}