use amfi::agent::{AgentIdentifier, InformationSet, ScoringInformationSet};
use amfi::domain::DomainParameters;
use crate::classic::common::{AsymmetricRewardTable, AsymmetricRewardTableInt};
use crate::classic::domain::{ClassicGameDomain, ClassicGameDomainNumbered, ClassicGameError, IntReward};
use crate::classic::domain::ClassicGameError::EncounterNotReported;
use crate::pairing::AgentNum;

#[derive(Copy, Clone, Debug)]
pub struct HistorylessInfoSet{
    id: AgentNum,
    reward_table: AsymmetricRewardTableInt,
    payoff: IntReward

}

impl HistorylessInfoSet{
    pub fn new(id: AgentNum, reward_table: AsymmetricRewardTableInt) -> Self{
        Self{
            id, reward_table, payoff: 0
        }
    }
}







impl InformationSet<ClassicGameDomain<AgentNum>> for HistorylessInfoSet{
    fn agent_id(&self) -> &AgentNum {
        &self.id
    }

    fn is_action_valid(&self, _action: &<ClassicGameDomain<AgentNum> as DomainParameters>::ActionType) -> bool {
        true
    }

    fn update(&mut self, update: <ClassicGameDomainNumbered as DomainParameters>::UpdateType) -> Result<(), ClassicGameError<AgentNum>> {

        if let Some(this_encounter_report) = update.get(self.id.clone() as usize){
            let reward = self.reward_table
                .reward_for_side(this_encounter_report.side, this_encounter_report.left_action(), this_encounter_report.right_action());

            self.payoff += reward;
            Ok(())
        } else{
            Err(EncounterNotReported(self.id as u32))
        }
            //.ok_or(Err(EncounterNotReported(self.id as u32)));




    }
}

impl ScoringInformationSet<ClassicGameDomainNumbered> for HistorylessInfoSet{
    type RewardType = IntReward;

    fn current_subjective_score(&self) -> Self::RewardType {
        self.payoff
    }

    fn penalty_for_illegal() -> Self::RewardType {
        -10
    }
}