#include<stdlib.h>
#include<cmath>
#include<chrono>
#include<thread>
#include<Eigen/Dense>
#include<librealsense2/h/rs_sensor.h>
#include<librealsense2/hpp/rs_pipeline.hpp>
#include<stack>
#include<tuple>

using namespace std;
using namespace Eigen;
	
class EKF
{
	public:
	Matrix<float, 4, 4> Q;
	Matrix<float, 2, 2> R;
	Matrix<float, 2, 2> ip_noise;
	Matrix<float, 2, 4> H;
	EKF()
	{
		//Covariance Matrix
        Q<<
         0.1, 0.0, 0.0, 0.0,
         0.0, 0.1, 0.0, 0.0,
         0.0, 0.0, 0.1, 0.0, 
	 0.0, 0.0, 0.0, 0.0; 

        R <<
     	   1,0,
         0,1;

        //input noise 
       ip_noise <<
        1.0, 0,
        0, (30*(3.14/180));

        //measurement matrix
       H<<
        1,0,0,0,
        0,1,0,0;
  	}

	float dt = 0.1;

	tuple<MatrixXf, MatrixXf> observation(MatrixXf xTrue, MatrixXf u)
	{	
		xTrue = state_model(xTrue, u);
		
		Matrix<float, 2, 1> ud;
		ud = u + (ip_noise * MatrixXf::Random(2,1));

		return make_tuple(xTrue, ud);
	}
	
	MatrixXf state_model(MatrixXf x, MatrixXf u)
	{
		Matrix<float, 4, 4> A;
	           A<< 1,0,0,0,
		    			0,1,0,0,
		    			0,0,1,0,
		    			0,0,0,0;

		Matrix<float, 4, 2> B;
		B<< (dt*cos(x.coeff(2,0))), 0,
		    (dt*sin(x.coeff(2,0))), 0,
		     0, dt,
		     1, 0;

		x = (A * x) + (B * u);

		return x;
	}			

	MatrixXf jacob_f(MatrixXf x, MatrixXf u)
	{	
		float yaw = x.coeff(2,0);

		float v = u.coeff(0,0);

		Matrix<float, 4, 4> jF;
		    jF<< 1.0, 0.0, (-dt*v*sin(yaw)), (dt*cos(yaw)),
		     		 0.0, 1.0, (dt*v*cos(yaw)), (dt*sin(yaw)),
		         0.0, 0.0, 1.0, 0.0,
		         0.0, 0.0, 0.0, 1.0;
		
		return jF;

	}

	MatrixXf observation_model(MatrixXf x)
	{
		Matrix<float, 2, 1> z;

		z = H * x;

		return z;
	}	

	tuple<MatrixXf, MatrixXf> ekf_estimation(MatrixXf xEst, MatrixXf PEst, MatrixXf z, MatrixXf u)
	{	
		//Predict 
		Matrix<float, 4, 1> xPred;
		xPred = state_model(xEst, u);

		//state vector
		Matrix<float, 4, 4> jF; 
		jF = jacob_f(xEst, u); 

		Matrix<float, 4, 4> PPred;
		PPred = (jF*PEst*jF.transpose()) + Q;

		//Update
		Matrix<float, 2, 1> zPred;
		zPred = observation_model(xPred);

		Matrix<float, 2, 1> y;
		y = z - zPred; //measurement residual 
		
		Matrix<float, 2, 2> S;
		S = (H*PPred*H.transpose()) + R; //Innovation Covariance
		
		Matrix<float, 4, 2> K;
		K = PPred*H.transpose()*S.inverse(); //Kalman Gain
		
		xEst = xPred + K * y; //update step

		PEst = (MatrixXf::Identity(4,4) - (K*H)) * PPred;

		return make_tuple(xEst, PEst);
	}
  	
};
	
int main()
{
	EKF obj;
  	
	//pipeline for realsens
	rs2::pipeline p;
	rs2::config c;

	c.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
	c.enable_stream(RS2_STREAM_GYRO,  RS2_FORMAT_MOTION_XYZ32F);	
	
	p.start(c);

	bool show_animation = false;
        bool print_to_cout = true;
	float accel_net = 0.0;
	
	//transofrmation matrix
	Matrix<float, 3, 3> rs2_to_base_tfm;	
	rs2_to_base_tfm<< 0,0,1,
        		  1,0,0,
	      		  0,1,0;
	
	Matrix<float, 1, 3> gyro;
	Matrix<float, 1, 3> accel;
 
        //state vector
        Matrix<float, 4, 1> xEst = MatrixXf::Zero(4,1);
	Matrix<float, 4, 1> xTrue = MatrixXf::Zero(4,1);
	Matrix<float, 4, 4> PEst = MatrixXf::Identity(4,4);
	Matrix<float, 2, 1> ud = MatrixXf::Zero(2,1);
	Matrix<float, 2, 1> z = MatrixXf::Zero(2,1);

	//history
	Matrix<float, 4, 1>  hxEst = MatrixXf::Zero(4,1);
        Matrix<float, 4, 1> hxTrue = MatrixXf::Zero(4,1);
	

	while (true)
	{
		auto frames = p.wait_for_frames();
		rs2::frame frame;
		if (frame = frames.first_or_default(RS2_STREAM_GYRO))
    		{
		    	auto motion = frame.as<rs2::motion_frame>();
        		rs2_vector gyro_data = motion.get_motion_data();
			gyro = {gyro_data.x, gyro_data.y, gyro_data.z};
			gyro = (rs2_to_base_tfm*gyro.transpose()).transpose();
		}

		if (frame = frames.first_or_default(RS2_STREAM_ACCEL))
    		{	
		    	auto motion = frame.as<rs2::motion_frame>();
        		rs2_vector accel_data = motion.get_motion_data();
			accel = {accel_data.x,accel_data.y, accel_data.z};
	                accel = (rs2_to_base_tfm * accel.transpose()).transpose();
   	        }      

		//calculating net acceleration
		float accel_norm = accel.norm();
		accel_net += accel_norm*obj.dt;

		//control input
		Matrix<float, 2, 1> u={accel_net, gyro(2)};
				
		float time = time + obj.dt;

		tie(xTrue, ud) = obj.observation(xTrue, u);
		
		z = obj.observation_model(xTrue);

		tie(xEst, PEst) = obj.ekf_estimation(xEst, PEst, z , ud);

		//store datat history
		hxEst = xEst;
		hxTrue = xTrue;

		if (print_to_cout) 
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			//Estimation + True
			cout <<hxEst(0) << " " << hxEst(1) <<" "<< hxTrue(0) << " " << hxTrue(1) <<endl;
			
		}
	}
    return 0;
}




	     	









