function dxdt = pendulumCT0(x, u)
%% Continuous-time nonlinear dynamic model of a pendulum
%
% 2 states (x): 
%   angle (theta): when 0, pendulum is at upright position
%   angular velocity (theta_dot): when positive, pendulum moves anti-clockwisely
% 
% 1 inputs: (u)
%   force (F): when positive, force pushes cart to right 
%
% Copyright 2018 The MathWorks, Inc.

%#codegen

%% parameters
mPend = 2.0;  % pendulum mass
g = 9.81;   % gravity of earth
L = 0.5;    % pendulum length
Kd = 0.2;    % damping
%% Obtain x, u and y
% x
theta = x(1);
theta_dot = x(2);
% u
F = u;
%% Compute dxdt
dxdt = [theta_dot;...
        -g/L*sin(theta) - Kd*L*theta_dot + 1/mPend*F];
