use std::cell::Cell;
use std::fmt::{Display, Formatter};
use rand::seq::IteratorRandom;
use sztorm::agent::Policy;
use sztorm::Reward;
use sztorm::state::agent::{InformationSet, ScoringInformationSet};
use crate::prisoner::common::RewardTable;
use crate::prisoner::domain::{PrisonerAction, PrisonerDomain, PrisonerError, PrisonerUpdate};
use crate::prisoner::domain::PrisonerAction::{Betray, Cover};

#[derive(Clone, Debug)]
pub struct PrisonerState{
    previous_actions: Vec<PrisonerUpdate>,
    reward_table: RewardTable,
    last_action: Cell<Option<PrisonerAction>>

}

impl PrisonerState{
    pub fn new(reward_table: RewardTable) -> Self{
        Self{reward_table, last_action: Cell::new(None), previous_actions: Vec::new()}
    }

    pub fn _select_action(&self, action: PrisonerAction){
        self.last_action.set(Some(action));
    }

    pub fn previous_actions(&self) -> &Vec<PrisonerUpdate>{
        &self.previous_actions
    }

    pub fn count_actions(&self, action: PrisonerAction) -> usize{
        self.previous_actions.iter().filter(|update| update.own_action == action)
            .count()
    }
}

impl Display for PrisonerState{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rounds: {} |", self.previous_actions.len())?;
        let mut s = self.previous_actions.iter().fold( String::new(),|mut acc, update|{
            acc.push_str(&format!("{:#}-{:#} ", update.own_action, update.other_prisoner_action));
            acc
        });
        s.pop();
        write!(f, "{}", s)
    }
}


pub struct CoverPolicy{}

impl Policy<PrisonerDomain> for CoverPolicy{
    type StateType = PrisonerState;

    fn select_action(&self, state: &Self::StateType) -> Option<PrisonerAction> {
        state._select_action(Cover);
        Some(Cover)
    }
}

pub struct Forgive1Policy{}

impl Policy<PrisonerDomain> for Forgive1Policy{
    type StateType = PrisonerState;

    fn select_action(&self, state: &Self::StateType) -> Option<PrisonerAction> {
        let enemy_betrayals = state.previous_actions().iter().filter(| &step|{
            step.other_prisoner_action == Betray
        }).count();
        if enemy_betrayals > 1 {
            state._select_action(Betray);
            Some(Betray)
        } else {
            state._select_action(Cover);
            Some(Cover)
        }

    }
}

pub struct BetrayRatioPolicy{}

impl Policy<PrisonerDomain> for BetrayRatioPolicy{
    type StateType = PrisonerState;

    fn select_action(&self, state: &Self::StateType) -> Option<PrisonerAction> {
        let betrayed = state.previous_actions().iter()
            .filter(|round| round.other_prisoner_action == Betray)
            .count();
        let covered = state.previous_actions().iter()
            .filter(|round| round.other_prisoner_action == Cover)
            .count();

        if betrayed > covered{
            state._select_action(Betray);
            Some(Betray)
        } else {
            state._select_action(Cover);
            Some(Cover)
        }
    }
}


pub struct RandomPrisonerPolicy{}


impl Policy<PrisonerDomain> for RandomPrisonerPolicy{
    type StateType = PrisonerState;


    fn select_action(&self, state: &Self::StateType) -> Option<PrisonerAction> {
        let mut rng = rand::thread_rng();
        state.available_actions().into_iter().choose(&mut rng).and_then(|a|{
            state._select_action(a);
            Some(a)
        })


    }
}



impl InformationSet<PrisonerDomain> for PrisonerState{
    type ActionIteratorType = [PrisonerAction;2];

    fn available_actions(&self) -> Self::ActionIteratorType {
        [Betray, Cover]
    }


    fn is_action_valid(&self, _action: &PrisonerAction) -> bool {
        true
    }

    fn update(&mut self, update: PrisonerUpdate) -> Result<(), PrisonerError> {
        let last = self.last_action.get();
        if let Some(my_action) = last{
            if my_action == update.own_action{
                self.previous_actions.push(update);
                self.last_action.set(None);
                Ok(())
            } else{
                Err(PrisonerError::DifferentActionPerformed {chosen: my_action, logged: update.own_action})
            }
        } else {
            Err(PrisonerError::NoLastAction(update.own_action))
        }
    }
}

impl ScoringInformationSet<PrisonerDomain> for PrisonerState{
    type RewardType = f64;

    fn current_subjective_score(&self) -> Self::RewardType {
        if !self.previous_actions.is_empty(){
            let sum = self.previous_actions.iter().fold(0.0, |acc, x|{
                acc + self.reward_table.reward(x.own_action, x.other_prisoner_action) as f64
            });
            sum/(self.previous_actions.len() as f64)

        } else{
            Self::RewardType::neutral()
        }



        //self.previous_actions.len() as f64
    }

    fn penalty_for_illegal() -> Self::RewardType {
        -100.0
    }
}

