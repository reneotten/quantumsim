e = 1.602*10^-19;   %C

eps_ox = 3.9;
eps_si = 11.2;
eps_0 = 8.85e-12;

V_ds = 1;   %V
V_g = 0;    %V

d_ch = 20;  %nm
d_ox = 1;    %nm
a = 0.01;       %nm
lambda = sqrt(eps_si/eps_ox * d_ch * d_ox); %nm
d_S  = 5 * lambda; %nm
d_D = d_S;          %nm

N_S = floor(d_S / a);
N_D = floor(d_D / a);
N_ch = floor(d_ch / a);
N = N_S + N_D + N_ch;

E_g = 1.12; %eV
E_f = 0;  %eV
phig = V_g; %eV

phi_g = zeros(N,1);
phi_g(N_S+1:(N_S+N_ch)) = phig;
phi_bi = zeros(N,1);
phi_bi(N_S+1:(N_S+N_ch)) = E_f + 0.5*E_g;
phi_bi((N_S+N_ch+1):N) = -V_ds;
rho = zeros(N,1);

ddx2 = eye(N) * (-2-1/lambda^2) + diag(ones(N-1,1),1) + diag(ones(N-1,1),-1);
ddx2(1,2) = 2;
ddx2(N,N-1) =2;
%ddx2 = ddx2./a^2;

vec_right = e/(eps_0*eps_si)*(rho)-(phi_g+phi_bi)/lambda^2;

phi_f = ddx2\vec_right;
plot(phi_f)
max(phi_f)