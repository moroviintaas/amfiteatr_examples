use std::fmt::Debug;
use amfi::agent::{InformationSet,
                  PresentPossibleActions,
                  ScoringInformationSet};
use amfi::domain::DomainParameters;
use amfi_classic::domain::{ClassicAction, ClassicGameDomainNumbered, IntReward};


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