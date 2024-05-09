#include<iostream>
#include<vector>
#include<chrono>
#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/bt_factory.h"

using namespace std::chrono_literals;

//AtWP
class AtWP : public BT::SyncActionNode
{
	public:
		explicit AtWP(const std::string &name) : BT::SyncActionNode(name, {})
			{}
		BT::NodeStatus tick() override
		{
			std::cout<<"Checking if Reached at WayPoint: "<<this->name() <<std::endl;
			int status;std::cin>>status;
			if(status){
				std::cout<<"Rover already at WayPoint"<<std::endl;
				return BT::NodeStatus::SUCCESS;
			}
			else{
				return BT::NodeStatus::FAILURE;
			}
		}
};

//GotoWP
class GotoWP : public BT::SyncActionNode
{
        public:
                explicit GotoWP(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Searching and Moving to WayPoint: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//TubeFound
class TubeFound : public BT::SyncActionNode
{
        public:
                explicit TubeFound(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Checking if the Tube is in sight: "<<this->name() <<std::endl;
			int status;std::cin>>status;
                        if(status){
                                std::cout<<"Tube is in Sight"<<std::endl;
                                return BT::NodeStatus::SUCCESS;
                        }
                        else{
                                return BT::NodeStatus::FAILURE;
                        }
                }
};

//SearchTube
class SearchTube : public BT::SyncActionNode
{
        public:
                explicit SearchTube(const std::string &name, const BT::NodeConfig &config) : BT::SyncActionNode(name, config)
                        {}

		static BT::PortsList providedPorts(){
    			return {BT::OutputPort<std::vector<double>>("TubePose")};
		}

                BT::NodeStatus tick() override
                {
                        std::cout<<"Searching for Tube\nTube Found: "<<this->name() <<std::endl;
			std::vector<double> pose{0.6,0.4,0.3};
			BT::TreeNode::setOutput("TubePose",pose);
                        return BT::NodeStatus::SUCCESS;
                }
};

//GotoTube
class GotoTube : public BT::SyncActionNode
{
        public:
                explicit GotoTube(const std::string &name, const BT::NodeConfig &config) : BT::SyncActionNode(name, config)
                        {}

		static BT::PortsList providedPorts(){
                        return {BT::InputPort<std::vector<double>>("TubePose")};
                }

                BT::NodeStatus tick() override
                {
			auto msg = getInput<std::vector<double>>("TubePose");

			if (!msg){
				throw BT::RuntimeError("missing required input[message]: ", msg.error());
			}

			for(const auto pval : msg.value()){
				std::cout<<pval<<' ';
			}

                        std::cout<<"Going towards Tube: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//OpenGripper
class OpenGripper : public BT::SyncActionNode
{
        public:
                explicit OpenGripper(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Opening the Gripper: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//CloseGripper
class CloseGripper : public BT::SyncActionNode
{
        public:
                explicit CloseGripper(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Closing the Gripper: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//ApproachTube using arm
class ApproachTube : public BT::SyncActionNode
{
        public:
                explicit ApproachTube(const std::string &name, const BT::NodeConfig &config) : BT::SyncActionNode(name, config)
                        {}
		
		static BT::PortsList providedPorts(){
                        return {BT::InputPort<std::vector<double>>("TubePose")};
                }

                BT::NodeStatus tick() override
                {
			auto msg = getInput<std::vector<double>>("TubePose");

                        if (!msg){
                                throw BT::RuntimeError("missing required input[message]: ", msg.error());
                        }

                        for(const auto pval : msg.value()){
                                std::cout<<pval<<' ';
                        }

                        std::cout<<"Approaching the Tube using Arm: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//GotoStorage using arm
class GotoStorage : public BT::SyncActionNode
{
        public:
                explicit GotoStorage(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Moving to Storage using Arm: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//AtFP
class AtFP : public BT::SyncActionNode
{
        public:
                explicit AtFP(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Checking if Final Point is in sight: "<<this->name() <<std::endl;
                        int status;std::cin>>status;
                        if(status){
                                std::cout<<"FP is in Sight"<<std::endl;
                                return BT::NodeStatus::SUCCESS;
                        }
                        else{
                                return BT::NodeStatus::FAILURE;
                        }
                }
};

//SearchFP
class SearchFP : public BT::SyncActionNode
{
        public:
                explicit SearchFP(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Rover Searches for FP while avoiding obstacles\nFP Found: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//TDNearby
class TDNearby : public BT::SyncActionNode
{
        public:
                explicit TDNearby(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Checking if Rover is near to Tube Drop: "<<this->name() <<std::endl;
                        int status;std::cin>>status;
                        if(status){
                                std::cout<<"Rover is near to Tube Drop"<<std::endl;
                                return BT::NodeStatus::SUCCESS;
                        }
                        else{
                                return BT::NodeStatus::FAILURE;
                        }
                }
};

//GotoTD
class GotoTD : public BT::SyncActionNode
{
        public:
                explicit GotoTD(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Rover moves near to the Tube Drop: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//PickTube
class PickTube : public BT::SyncActionNode
{
        public:
                explicit PickTube(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Arm Picks Tube from the Storage: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//PlaceTube
class PlaceTube : public BT::SyncActionNode
{
        public:
                explicit PlaceTube(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Arm places the tube in the Tube Drop: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};

//Park
class Park : public BT::SyncActionNode
{
        public:
                explicit Park(const std::string &name) : BT::SyncActionNode(name, {})
                        {}
                BT::NodeStatus tick() override
                {
                        std::cout<<"Rover Parks near to the Final Point: "<<this->name() <<std::endl;
                        return BT::NodeStatus::SUCCESS;
                }
};




int main()
{
	BT::BehaviorTreeFactory factory;
	factory.registerNodeType<AtWP>("AtWP");
	factory.registerNodeType<GotoWP>("GotoWP");
	factory.registerNodeType<TubeFound>("TubeFound");
	factory.registerNodeType<SearchTube>("SearchTube");
	factory.registerNodeType<GotoTube>("GotoTube");
	factory.registerNodeType<OpenGripper>("OpenGripper");
	factory.registerNodeType<ApproachTube>("ApproachTube");
	factory.registerNodeType<CloseGripper>("CloseGripper");
	factory.registerNodeType<GotoStorage>("GotoStorage");
	factory.registerNodeType<AtFP>("AtFP");
	factory.registerNodeType<SearchFP>("SearchFP");
	factory.registerNodeType<TDNearby>("TDNearby");
	factory.registerNodeType<GotoTD>("GotoTD");
	factory.registerNodeType<PickTube>("PickTube");
	factory.registerNodeType<PlaceTube>("PlaceTube");
	factory.registerNodeType<Park>("Park");

	auto tree = factory.createTreeFromFile("./bt_tree.xml");

	tree.tickWhileRunning();
	return 0;
}
