use crate::prisoner::domain::{PrisonerAction, PrisonerReward};

#[derive(Debug, Copy, Clone)]
pub struct RewardTable{
    pub cover_v_cover: PrisonerReward,
    pub cover_v_betray: PrisonerReward,
    pub betray_v_cover: PrisonerReward,
    pub betray_v_betray: PrisonerReward
}

impl RewardTable{

    pub fn reward(&self, action: PrisonerAction, other_action: PrisonerAction) -> PrisonerReward{

        match (action, other_action){
            (PrisonerAction::Cover, PrisonerAction::Cover) => self.cover_v_cover,
            (PrisonerAction::Cover, PrisonerAction::Betray) => self.cover_v_betray,
            (PrisonerAction::Betray, PrisonerAction::Cover) => self.betray_v_cover,
            (PrisonerAction::Betray, PrisonerAction::Betray) => self.betray_v_betray
        }
    }

}