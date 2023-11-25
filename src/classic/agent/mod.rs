mod prisoner;
mod historyless;
mod own_history;


use std::fmt::{Debug, Formatter, Pointer};
use amfi::agent::{AgentIdentifier, InformationSet, PresentPossibleActions, ScoringInformationSet};
use amfi::domain::DomainParameters;
pub use prisoner::*;
pub use historyless::*;
use crate::classic::domain::{ClassicAction, ClassicGameDomainNumbered, IntReward};
use crate::pairing::AgentNum;
pub use own_history::*;


#[derive(Debug)]
pub struct BoxedClassicInfoSet{
    pub internal: Box<dyn ScoringInformationSet<ClassicGameDomainNumbered,  RewardType=IntReward>>,
}

impl InformationSet<ClassicGameDomainNumbered> for BoxedClassicInfoSet {
    fn agent_id(&self) -> &<ClassicGameDomainNumbered as DomainParameters>::AgentId {
        self.internal.agent_id()
    }

    fn is_action_valid(&self, action: &<ClassicGameDomainNumbered as DomainParameters>::ActionType) -> bool {
        self.internal.is_action_valid(action)
    }

    fn update(&mut self, update: <ClassicGameDomainNumbered as DomainParameters>::UpdateType) -> Result<(), <ClassicGameDomainNumbered as DomainParameters>::GameErrorType> {
        self.internal.update(update)
    }
}


impl ScoringInformationSet<ClassicGameDomainNumbered> for BoxedClassicInfoSet{
    type RewardType = IntReward;

    fn current_subjective_score(&self) -> Self::RewardType {
        self.internal.current_subjective_score()
    }

    fn penalty_for_illegal(&self) -> Self::RewardType {
        self.internal.penalty_for_illegal()
    }
}



impl PresentPossibleActions<ClassicGameDomainNumbered> for BoxedClassicInfoSet{
    type ActionIteratorType = [ClassicAction;2];

    fn available_actions(&self) -> Self::ActionIteratorType {
        [ClassicAction::Cooperate, ClassicAction::Defect]
    }
}