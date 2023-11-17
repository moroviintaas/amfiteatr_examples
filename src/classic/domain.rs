use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use amfi::agent::{AgentIdentifier};
use amfi::error::{AmfiError};
use amfi::domain::{Action, DomainParameters};
use crate::classic::domain::PrisonerId::{Andrzej, Janusz};
use enum_map::Enum;
use serde::{Deserialize, Serialize};
use crate::classic::common::Side;
use crate::pairing::AgentNum;

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
pub enum ClassicGameError<ID: AgentIdentifier> {
    #[error("Performed different action (chosen: {chosen:?}, logged: {logged:?})")]
    DifferentActionPerformed{
        chosen: ClassicAction,
        logged: ClassicAction
    },
    #[error("Order in game was violated. Current player given by current_player(): {0} was rejected in forward()")]
    ViolatedOrder(ID),
    #[error("Environment logged action {0}, but none was performed")]
    NoLastAction(ClassicAction),
    #[error("Player: {0} played after GameOver")]
    ActionAfterGameOver(ID),
    #[error("Player: {0} played out of order")]
    ActionOutOfOrder(ID),
    #[error("Value can't be probability: {0}")]
    NotAProbability(f64),
    #[error("Odd number of players: {0}")]
    ExpectedEvenNumberOfPlayers(u32),
    #[error("Update does no include requested encounter report for agent: {0}")]
    EncounterNotReported(AgentNum),
}

/*
impl Into<AmfiError<PrisonerDomain>> for PrisonerError {
    fn into(self) -> AmfiError<PrisonerDomain> {
        AmfiError::Game(self)
    }
}

 */
impl<ID: AgentIdentifier> From<ClassicGameError<ID>> for AmfiError<ClassicGameDomain<ID>>{
    fn from(value: ClassicGameError<ID>) -> Self {
        AmfiError::Game(value)
    }
}


#[derive(Clone, Debug)]
pub struct ClassicGameDomain<ID: AgentIdentifier>{
    _id: PhantomData<ID>
}

#[derive(Debug, Copy, Clone)]
pub struct EncounterReport<ID: AgentIdentifier> {

    pub own_action: ClassicAction,
    pub other_player_action: ClassicAction,
    pub side: Side,
    pub other_id: ID,

}

impl<ID: AgentIdentifier> EncounterReport<ID>{
    pub fn left_action(&self) -> ClassicAction{
        match self.side{
            Side::Left => self.own_action,
            Side::Right => self.other_player_action
        }
    }
    pub fn right_action(&self) -> ClassicAction{
        match self.side{
            Side::Left => self.other_player_action,
            Side::Right => self.own_action
        }
    }
    pub fn side_action(&self, side: Side) -> ClassicAction{
        match side{
            Side::Left => self.left_action(),
            Side::Right => self.right_action(),
        }
    }
}

pub type EncounterReportNamed = EncounterReport<PrisonerId>;
pub type EncounterReportNumbered = EncounterReport<AgentNum>;

impl<ID: AgentIdentifier> Display for EncounterReport<ID> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Update [own action: {}, opponent's action: {}]", self.own_action, self.other_player_action)
    }
}

//impl StateUpdate for PrisonerUpdate{}

//pub type PrisonerId = u8;
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Enum)]
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


impl<ID: AgentIdentifier> DomainParameters for ClassicGameDomain<ID> {
    type ActionType = ClassicAction;
    type GameErrorType = ClassicGameError<ID>;
    type UpdateType = Arc<Vec<EncounterReport<ID>>>;
    type AgentId = ID;
    type UniversalReward = IntReward;
}
pub type ClassicGameDomainNamed = ClassicGameDomain<PrisonerId>;
pub type ClassicGameDomainNumbered = ClassicGameDomain<AgentNum>;
