use crate::classic::domain::{ClassicAction, IntReward};
use serde::{Serialize, Deserialize};
use amfi::domain::Reward;
use enum_map::{Enum, enum_map, EnumMap};

#[derive(Debug, Copy, Clone, Enum, Serialize, Deserialize)]
pub enum Side{
    Left,
    Right
}

impl Default for Side{
    fn default() -> Self {
        Self::Left
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub struct SymmetricRewardTable<R: Reward + Copy> {

    la: EnumMap<ClassicAction, EnumMap<ClassicAction, R>>

    //pub coop_when_coop: R,
    //pub coop_when_defect: R,
    //pub defect_when_coop: R,
    //pub defect_when_defect: R
}

pub type SymmetricRewardTableInt = SymmetricRewardTable<IntReward>;


impl<R: Reward + Copy> SymmetricRewardTable<R> {

    pub fn new(coop_when_coop: R, coop_when_defect: R, defect_when_coop: R, defect_when_defect: R) -> Self{
        Self{
            la: enum_map! {
                ClassicAction::Defect => enum_map! {
                    ClassicAction::Defect => defect_when_defect,
                    ClassicAction::Cooperate => defect_when_coop,
                },
                ClassicAction::Cooperate => enum_map! {
                    ClassicAction::Defect => coop_when_defect,
                    ClassicAction::Cooperate => coop_when_coop,
                }
            }
        }
    }

    pub fn reward(&self, action: ClassicAction, other_action: ClassicAction) -> R {
        /*
        match (action, other_action){
            (ClassicAction::Cooperate, ClassicAction::Cooperate) => &self.coop_when_coop,
            (ClassicAction::Cooperate, ClassicAction::Defect) => &self.coop_when_defect,
            (ClassicAction::Defect, ClassicAction::Cooperate) => &self.defect_when_coop,
            (ClassicAction::Defect, ClassicAction::Defect) => &self.defect_when_defect
        }

         */
        self.la[action][other_action]
    }

}


#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct AsymmetricRewardTable<R: Reward + Copy>{

    table: EnumMap<Side, SymmetricRewardTable<R>>
}

pub type AsymmetricRewardTableInt = AsymmetricRewardTable<IntReward>;

impl<R: Reward + Copy> AsymmetricRewardTable<R> {

    pub fn new(left_table: SymmetricRewardTable<R>, right_table: SymmetricRewardTable<R>) -> Self{
        Self{
            table: enum_map! {
                Side::Left => left_table,
                Side::Right => right_table
            }
        }
    }

    pub fn reward_for_side(&self, reward_for: Side, left_action: ClassicAction, right_action: ClassicAction) -> R {

        self.table[reward_for].reward(left_action, right_action)
    }

    pub fn rewards(&self, left_action: ClassicAction, right_action: ClassicAction) -> (R, R){
        (
            self.table[Side::Left].reward(left_action, right_action),
            self.table[Side::Right].reward(left_action, right_action)
        )
    }



}

impl<R: Reward + Copy> From<SymmetricRewardTable<R>> for AsymmetricRewardTable<R>{
    fn from(value: SymmetricRewardTable<R>) -> Self {
        AsymmetricRewardTable::new(value, value)
    }
}



#[cfg(test)]
mod tests{
    use std::mem::size_of;
    use crate::classic::common::{AsymmetricRewardTableInt, SymmetricRewardTableInt};

    #[test]
    fn size_of_symmetric_table(){
        assert_eq!(size_of::<SymmetricRewardTableInt>(), 16);
    }
    #[test]
    fn size_of_asymmetric_table(){
        assert_eq!(size_of::<AsymmetricRewardTableInt>(), 32);
    }
}


