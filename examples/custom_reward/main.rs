use std::{default, thread};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use log::LevelFilter;
use tch::{Device, nn, Tensor};
use tch::nn::{Adam, VarStore};
use amfi_examples::classic::agent::{OwnHistoryInfoSet, OwnHistoryTensorRepr, PrisonerInfoSet, PrisonerInfoSetWay};
use amfi_examples::classic::common::{SymmetricRewardTable, SymmetricRewardTableInt};
use amfi_examples::classic::domain::{ClassicAction, ClassicGameDomain, ClassicGameDomainNumbered, UsizeAgentId};
use amfi_examples::pairing::{AgentNum, PairingState};
use amfi_rl::tensor_repr::{ConvertToTensor, WayToTensor};
use amfi_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorA2C};
use clap::Parser;
use amfi::agent::{AgentGenT, AutomaticAgentRewarded, TracingAgent};
use amfi::comm::EnvMpscPort;
use amfi::env::{AutoEnvironmentWithScores, ScoreEnvironment, TracingEnv};
use amfi::env::generic::{BasicEnvironment, TracingEnvironment};
use amfi_examples::classic::policy::ClassicPureStrategy;
use amfi_rl::actor_critic::ActorCriticPolicy;
use amfi_rl::TrainConfig;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct EducatorOptions{

    #[arg(short = 'v', long = "log_level", value_enum, default_value = "debug")]
    pub log_level: LevelFilter,

    #[arg(short = 'a', long = "log_level_amfi", value_enum, default_value = "OFF")]
    pub log_level_amfi: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 's', long = "save")]
    pub save_file: Option<PathBuf>,

    #[arg(short = 'l', long = "load")]
    pub load_file: Option<PathBuf>,

    #[arg(short = 'e', long = "epochs", default_value = "10")]
    pub epochs: usize,

    #[arg(short = 'n', long = "rounds", default_value = "32")]
    pub number_of_rounds: usize


    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}

pub struct ModelElements<ID: UsizeAgentId, Seed>{
    environment: Arc<Mutex<dyn AutoEnvironmentWithScores<ClassicGameDomain<ID>>>>,
    agents: [Arc<Mutex<dyn AutomaticAgentRewarded<ClassicGameDomain<ID>>>>;2],
    seed: PhantomData<Seed>,
}

pub fn setup_logger(options: &EducatorOptions) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        //.level(options.log_level)
        .level_for("amfi_examples", options.log_level)
        //.level_for("pairing", options.log_level)
        //.level_for("classic", options.log_level);
        .level_for("amfi", options.log_level_amfi);

        match &options.log_file{
            None => dispatch.chain(std::io::stdout()),
            Some(f) => dispatch.chain(fern::log_file(f)?)
        }

        .apply()?;
    Ok(())
}


fn main() {

    let args = EducatorOptions::parse();
    let device = Device::Cpu;
    type Domain = ClassicGameDomainNumbered;
    let number_of_players = 2;

    let tensor_repr = OwnHistoryTensorRepr::new(args.number_of_rounds);

    let input_size = tensor_repr.desired_shape().iter().product();
    let normal_var_store = VarStore::new(device);
    let custom_var_store = VarStore::new(device);

    let mut env_adapter = EnvMpscPort::new();
    let comm0 = env_adapter.register_agent(0).unwrap();
    let comm1 = env_adapter.register_agent(1).unwrap();

    let reward_table = SymmetricRewardTableInt::new(5, 1, 10, 3);


    let net_template = NeuralNetTemplate::new(|path|{
        let seq = nn::seq()
            .add(nn::linear(path / "input", input_size, 512, Default::default()))
            .add(nn::linear(path / "h1", 512, 512, Default::default()));
            //.add(nn::linear(path / "h2", 512, 512, Default::default()));
        let actor = nn::linear(path / "al", 512, 2, Default::default());
        let critic =  nn::linear(path / "ac", 512, 1, Default::default());
        {move |input: &Tensor|{
            let xs = input.to_device(device).apply(&seq);
            TensorA2C{critic: xs.apply(&critic), actor: xs.apply(&actor)}
        }}
    });




    //let normal_policy =


    /*
    let net = A2CNet::new()
    let net  = nn::seq()
        .add(nn::linear(path / "input"))

     */
    //let state_template = PairingState::<AgentNum>::new_even(2, 10, reward_table.into()).unwrap();
    let env_state_template = PairingState::new_even(number_of_players, args.number_of_rounds, reward_table.into()).unwrap();
    let mut environment = TracingEnvironment::new(env_state_template.clone(), env_adapter);


    let net0 = A2CNet::new(VarStore::new(device), net_template.get_net_closure());
    let opt0 = net0.build_optimizer(Adam::default(), 1e-4).unwrap();
    let normal_policy = ActorCriticPolicy::new(net0, opt0, tensor_repr, TrainConfig {gamma: 0.99});
    let state0 = OwnHistoryInfoSet::new(0, reward_table.into());
    let mut normal_agent = AgentGenT::new(state0, comm0, normal_policy);


    let state1 = OwnHistoryInfoSet::new(1, reward_table.into());
    let test_policy = ClassicPureStrategy::new(ClassicAction::Defect);

    let mut test_agent = AgentGenT::new(state1, comm1, test_policy);


    //evaluate on start

    thread::scope(|s|{
        s.spawn(||{
            environment.run_with_scores().unwrap();
        });
        s.spawn(||{
            normal_agent.run_rewarded().unwrap()
        });
        s.spawn(||{
            test_agent.run_rewarded().unwrap()
        });
    });

    println!("{}", normal_agent.game_trajectory().list().last().unwrap());



    println!("{}", environment.trajectory().list().last().unwrap());

    println!("Scores: 0: {},\t1: {}", environment.actual_score_of_player(&0), environment.actual_score_of_player(&1));



    //let standard_strategy =
}