use std::cell::Cell;
use std::fmt::{Display, Formatter};
use rand::seq::IteratorRandom;
use tch::Tensor;
use sztorm::agent::Policy;
use sztorm::error::ConvertError;
use sztorm::Reward;
use sztorm::state::agent::{InformationSet, ScoringInformationSet};
use sztorm_rl::tensor_repr::{ActionTensor, ConvStateToTensor};
use crate::prisoner::common::RewardTable;
use crate::prisoner::domain::{PrisonerAction, PrisonerDomain, PrisonerError, PrisonerUpdate};
use crate::prisoner::domain::PrisonerAction::{Betray, Cover};

#[derive(Clone, Debug)]
pub struct PrisonerState{
    previous_actions: Vec<PrisonerUpdate>,
    reward_table: RewardTable,
    //last_action: Cell<Option<PrisonerAction>>

}

impl PrisonerState{
    pub fn new(reward_table: RewardTable) -> Self{
        Self{
            reward_table,
            //last_action: Cell::new(None),
            previous_actions: Vec::new()}
    }

    /*pub fn _select_action(&self, action: PrisonerAction){
        self.last_action.set(Some(action));
    }

     */

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
        //state._select_action(Cover);
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
            //state._select_action(Betray);
            Some(Betray)
        } else {
            //state._select_action(Cover);
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
            //state._select_action(Betray);
            Some(Betray)
        } else {
            //state._select_action(Cover);
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
            //state._select_action(a);
            Some(a)
        })


    }
}


pub struct SwitchOnTwoSubsequent{}

impl Policy<PrisonerDomain> for SwitchOnTwoSubsequent{
    type StateType = PrisonerState;

    fn select_action(&self, state: &Self::StateType) -> Option<PrisonerAction> {

        if let Some(i_update) = state.previous_actions().last(){
            let mut other_action = i_update.other_prisoner_action;
            for i in (0..state.previous_actions.len()-1).rev(){
                if state.previous_actions()[i].other_prisoner_action == other_action{
                    return Some(other_action)
                } else {
                    other_action = state.previous_actions()[i].other_prisoner_action;
                }
            }
            Some(Cover)
        } else{
            Some(Cover)
        }

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
        /*
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

         */
        self.previous_actions.push(update);
        Ok(())
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

pub struct PrisonerStateTranslate{

}

impl ConvStateToTensor<PrisonerState> for PrisonerStateTranslate{
    fn make_tensor(&self, t: &PrisonerState) -> Tensor {
        let mut array = [0.0f32;2*256];
        for i in 0..t.previous_actions().len(){
            array[2*i] = match t.previous_actions()[i].own_action{
                Betray =>  1.0,
                Cover => 2.0,
            };
            array[2*i+1] = match t.previous_actions()[i].other_prisoner_action{
                Betray =>  1.0,
                Cover => 2.0,
            };
        }
        Tensor::from_slice(&array[..])
    }
}

impl ActionTensor for PrisonerAction{
    fn to_tensor(&self) -> Tensor {
        match self{
            Betray => Tensor::from_slice(&[1.0f64;1]),
            Cover => Tensor::from_slice(&[2.0f64;1])
        }
    }


    /// ```
    /// use tch::Tensor;
    /// use sztorm_examples::prisoner::domain::PrisonerAction;
    /// use sztorm_examples::prisoner::domain::PrisonerAction::{Betray, Cover};
    /// use sztorm_rl::tensor_repr::ActionTensor;
    /// let t = Tensor::from_slice(&[1i64;1]);
    /// let action = PrisonerAction::try_from_tensor(&t).unwrap();
    /// assert_eq!(action, Cover);
    /// ```
    fn try_from_tensor(t: &Tensor) -> Result<Self, ConvertError> {


        let v: Vec<i64> = match Vec::try_from(t){
            Ok(v) => v,
            Err(e) =>{
                return Err(ConvertError::ActionDeserialize(format!("{}", t)))
            }
        };
        match v[0]{
            0 => Ok(Betray),
            1 => Ok(Cover),
            _ => Err(ConvertError::ActionDeserialize(format!("{}", t)))
        }
    }
}