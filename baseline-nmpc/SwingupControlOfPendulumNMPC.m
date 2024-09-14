%% Swing-up Control of a Pendulum Using Nonlinear Model Predictive Control
% This example uses a nonlinear model predictive controller object and
% block to achieve swing-up and balancing control of an inverted pendulum
% on a cart.

% Copyright 2016-2020 The MathWorks, Inc.

%% Product Requirement
% This example requires Optimization Toolbox(TM) software to provide the
% default nonlinear programming solver for nonlinear MPC to compute optimal
% control moves at each control interval.
if ~mpcchecktoolboxinstalled('optim')
    disp('Optimization Toolbox is required to run this example.')
    return
end

m = 2.0;
l = 0.5;
g = 9.81;
c = 0.2;

%% Pendulum/Cart Assembly
% The plant for this example is a pendulum/cart assembly, where _z_ is the
% cart position and _theta_ is the pendulum angle. The manipulated variable
% for this system is a variable force _F_ acting on the cart. The range of
% the force is between |-100| and |100|. An impulsive disturbance _dF_ can
% push the pendulum as well.
%
% <<../pendulumDiagramNMPC.png>>

%% Control Objectives
% Assume the following initial conditions for the pendulum/cart assembly.
%
% * The cart is stationary at _z_ = |0|.
% * The pendulum is in a downward equilibrium position where _theta_ =
% |-pi|.
%
% The control objectives are:
%
% * Swing-up control: Initially swing the pendulum up to an inverted
% equilibrium position where _z_ = |0| and _theta_ = |0|.
% * Cart position reference tracking: Move the cart to a new position with
% a step setpoint change, keeping the pendulum inverted.
% * Pendulum balancing: When an impulse disturbance of magnitude of |2|
% is applied to the inverted pendulum, keep the pendulum balanced, and
% return the cart to its original position.
%
% The downward equilibrium position is stable, and the inverted equilibrium
% position is unstable, which makes swing-up control more challenging for a
% single linear controller, which nonlinear MPC handles easily.

%% Control Structure
% In this example, the nonlinear MPC controller has the following I/O
% configuration.
%
% * One manipulated variable: Variable force (_F_)
% * Two measured outputs: Cart position (_z_) and pendulum angle
% (_theta_)
%
% Two other states, cart velocity (_zdot_) and pendulum angular velocity
% (_thetadot_) are not measurable.
%
% While the setpoint of the cart position, _z_, can vary, the setpoint of
% the pendulum angle, _theta_, is always |0| (inverted equilibrium
% position).

%% Create Nonlinear MPC Controller
% Create a nonlinear MPC controller with the proper dimensions using an
% <docid:mpc_ref#mw_56547a96-970b-449a-929e-a3dadc671ab4> object. In this
% example, the prediction model has |4| states, |2| outputs, and |1| input
% (MV).
nx = 2;
ny = 1;
nu = 1;
nlobj = nlmpc(nx, ny, nu);

%%
% The prediction model has a sample time of |0.1| seconds, which is the
% same as the controller sample time.
Ts = 0.2;
nlobj.Ts = Ts;

%%
% Set the prediction horizon to |10|, which is long enough to capture major
% dynamics in the plant but not so long that it hurts computational
% efficiency. 
nlobj.PredictionHorizon = 10;

%%
% Set the control horizon to |5|, which is long enough to give the
% controller enough degrees of freedom to handle the unstable mode without
% introducing excessive decision variables.
nlobj.ControlHorizon = 10;

%% Specify Nonlinear Plant Model
% The major benefit of nonlinear model predictive control is that it uses a
% nonlinear dynamic model to predict plant behavior in the future across a
% wide range of operating conditions.
%
% This nonlinear model is usually a first principle model consisting of a
% set of differential and algebraic equations (DAEs). In this example, a
% discrete-time cart and pendulum system is defined in the |pendulumDT0|
% function. This function integrates the continuous-time model,
% |pendulumCT0|, between control intervals using a multistep forward Euler
% method. The same function is also used by the nonlinear state estimator.
nlobj.Model.StateFcn = "pendulumDT0";

%%
% To use a discrete-time model, set the |Model.IsContinuousTime| property
% of the controller to |false|.
nlobj.Model.IsContinuousTime = false;

%%
% The prediction model uses an optional parameter, |Ts|, to represent the
% sample time. Using this parameter means that, if you change the
% prediction sample time during the design, you do not have to modify the
% |pendulumDT0| file.
nlobj.Model.NumberOfParameters = 1;

%%
% The two plant outputs are the first and third state in the model, the
% cart position and pendulum angle, respectively. The corresponding
% output function is defined in the |pendulumOutputFcn| function.
nlobj.Model.OutputFcn = 'pendulumOutputFcn';

%%
% It is best practice to provide analytical Jacobian functions whenever
% possible, since they significantly improve the simulation speed. In this
% example, provide a Jacobian for the output function using an anonymous
% function.
nlobj.Jacobian.OutputFcn = @(x,u,Ts) [1 0];

%%
% Since you do not provide Jacobian for the state function, the nonlinear
% MPC controller estimates the state function Jacobian during optimization
% using numerical perturbation. Doing so slows down simulation to some
% degree.

%% Define Cost and Constraints
% Like linear MPC, nonlinear MPC solves a constrained optimization problem
% at each control interval. However, since the plant model is nonlinear,
% nonlinear MPC converts the optimal control problem into a nonlinear
% optimization problem with a nonlinear cost function and nonlinear
% constraints.
%
% The cost function used in this example is the same standard cost function
% used by linear MPC, where output reference tracking and manipulated
% variable move suppression are enforced. Therefore, specify standard MPC
% tuning weights.
nlobj.Weights.OutputVariables = [1];
nlobj.Weights.ManipulatedVariablesRate = 0.1;

%%
% The force has a range between |-100| and |100|.
nlobj.MV.Min = -30;
nlobj.MV.Max = 30;

%% Validate Nonlinear MPC Controller
% After designing a nonlinear MPC controller object, it is best practice to
% check the functions you defined for the prediction model, state function,
% output function, custom cost, and custom constraints, as well as their
% Jacobians. To do so, use the |validateFcns| command. This function
% detects any dimensional and numerical inconsistencies in these functions.
theta0 = 0.0;
theta_dot0 = 0.0;
x0 = [theta0; theta_dot0];
u0 = 0.0;
validateFcns(nlobj,x0,u0,[],{Ts});

%% State Estimation
% In this example, only two plant states (cart position and pendulum angle)
% are measurable. Therefore, you estimate the four plant states using an
% extended Kalman filter. Its state transition function is defined in
% |pendulumStateFcn.m| and its measurement function is defined in
% |pendulumMeasurementFcn.m|.
EKF = extendedKalmanFilter(@pendulumStateFcn, @pendulumMeasurementFcn);

%% Closed-Loop Simulation in MATLAB(R)
% Specify the initial conditions for simulations by setting the initial
% plant state and output values. Also, specify the initial state of the
% extended Kalman filter.
%
% The initial conditions of the simulation areas follows.
%
% * The cart is stationary at _z_ = 0.
% * The pendulum is in a downward equilibrium position, _theta_ = |-pi|.
%
x = [0.0; 0.0];
y = [x(1)];
EKF.State = x;

%%
% |mv| is the optimal control move computed at any control interval.
% Initialize |mv| to zero, since the force applied to the cart is zero at
% the beginning.
mv = 0;

%%
% In the first stage of the simulation, the pendulum swings up from a
% downward equilibrium position to an inverted equilibrium position. The
% state references for this stage are all zero.
yref = [pi];


%%
% Using the <docid:mpc_ref#mw_953dc5ee-1b69-4a61-92d3-b6ae8a304f74>
% command, compute optimal control moves at each control interval. This
% function constructs a nonlinear programming problem and solves it using
% the |fmincon| function from the Optimization Toolbox.
%
% Specify the prediction model parameter using an
% <docid:mpc_ref#mw_346dc978-860c-4a9a-a25a-591184c50b61> object, and pass
% this object to |nlmpcmove|.
nloptions = nlmpcmoveopt;
nloptions.Parameters = {Ts};

%%
% Run the simulation for |20| seconds.
Duration = 20;
hbar = waitbar(0,'Simulation Progress');
x_NMPC = x;
y_NMPC = y;
u_NMPC = mv;
for ct = 1:(20/Ts)
    
    % Correct previous prediction using current measurement.
    xk = correct(EKF, y);
    % Compute optimal control moves.
    [mv,nloptions,info] = nlmpcmove(nlobj,xk,mv,yref,[],nloptions);
    % Predict prediction model states for the next iteration.
    predict(EKF, [mv; Ts]);
    % Implement first optimal control move and update plant states.
    x = pendulumDT0(x,mv,Ts);
    % Generate sensor data with some white noise.
    y = x(1) + randn(1)*0.01; 
    % Save plant states for display.
    y_NMPC = [y_NMPC y];
    x_NMPC = [x_NMPC x];
    u_NMPC = [u_NMPC mv];
    waitbar(ct*Ts/20,hbar);
end
close(hbar)

%%
% Plot the closed-loop response.
figure
subplot(1,2,1)
plot(0:Ts:Duration,x_NMPC(1,:))
xlabel('time')
ylabel('theta')
title('pendulum angle')
subplot(1,2,2)
plot(0:Ts:Duration,x_NMPC(2,:))
xlabel('time')
ylabel('thetadot')
title('pendulum velocity')

%%

figure
subplot(1,2,1)
plot(0:Ts:Duration,y_NMPC)
xlabel('time')
ylabel('theta')
subplot(1,2,2)
plot(0:Ts:Duration,u_NMPC)
xlabel('time')
ylabel('torque')

%%

save('results/NMPC.mat', 'x_NMPC', 'y_NMPC', 'u_NMPC'); 

