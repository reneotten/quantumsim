%% constants, lenth in nm, energy in eV
N=171;          % # of grid points

E_f=0.05;       % fermi energy in eV
E_g=1;        % band gap in eV
V_ds=0.5;       % drain-source voltage in V
V_g=0;          % gate potential Psi_g=-e*V_g in eV
d_ox=5;         % oxide thickness in nm
d_ch=5;        % channel thickness in nm
e=util.const.e; % elementary charge 
k_0=8.85e-12;   % dielectric constant
k_Si=11.2;     % dielectric constant for Silicon
k_ox=3.9;       % dielectric constant oxide
geo=1;          % Geometriefaktor # of gates, 'w' wrapgate
l_ch=30;        % channel length
l_ds=40;        % length of drain and source regions
N_dot=0;        % # Dopands


lambda=sqrt(k_Si/k_ox*d_ch*d_ox/geo); 

%% calculate source, drain and channel regions

S=[];
for l_ch=5:50
% calc overall length
l_g=2*l_ds+l_ch;
% calculate spacing
a=l_g/(N-1);
% calculate indices
n_left=floor(l_ds/a);   % source region 1 to n-left
n_right=N-n_left;       % drain region n_right-end

% create sparse laplacian
tic;
super=zeros(N,1);
super(2:end)=1;
super(2)=2;
sub=zeros(N,1);
sub(1:end-1)=1;
sub(end-1)=2;
middle=zeros(N,1);
middle(:)=-2;
L_sparse=spdiags([super,middle,sub],[1,0,-1],N,N)/a.^2;
toc;


I=[];
for V_g=0:0.01:1
% assign Psi_g and Psi_b for all regions
%create empty arrays
Psi_g=zeros(N,1);
Psi_bi=zeros(N,1);

% soure both 0

% channel
Psi_g(n_left+1:n_right-1) =-V_g;
Psi_bi(n_left+1:n_right-1)=E_f+E_g/2.;

%drain
Psi_g(n_right:end) =0;
Psi_bi(n_right:end)=-V_ds;

% create charge density
rho=zeros(N,1);

% solve eq. for Psi_f
% solve sparse (much quicker!)
tic;
Psi_f=(L_sparse-spdiags(zeros(N,1)+1/lambda^2,0,N,N))\((rho+N_dot)/k_0/k_Si-1/lambda^2*(Psi_g+Psi_bi));
toc;
% Plot
%figure, plot((0:N-1).*a,Psi_f);
Psi_0=max(Psi_f);

% 2nd Lecture
epsilon=10e-15;         % tolarance for fermi function
E_fs=0.05;  %?????
E_fd=-V_ds+0.05;
T=300;  %temperature in Kelvin
dE=0.001; % energy step

E_max=E_fs-util.const.k_b*T*log(epsilon)/util.const.e;
E=Psi_0:dE:E_max;

f_s = @(E) 1./(exp((E-E_fs).*util.const.e./util.const.k_b./T)+1);
f_d = @(E) 1./(exp((E-E_fd).*util.const.e./util.const.k_b./T)+1);

I(end+1)=2*util.const.e/util.const.h*sum(f_s(E)-f_d(E))*dE*e*1e-3;
end

x=0:0.01:1;
y=log10(I);
index = (x >= 0) & (x <= 0.4);
p = polyfit(x(index),y(index),1);  %# Fit polynomial coefficients for line
yfit = p(2)+x.*p(1);  %# Compute the best-fit line
%figure, plot(x,y);            %# Plot the data
%hold on;              %# Add to the plot
%close all
%plot(x,yfit,'r');     %# Plot the best-fit line
S(end+1)=1/p(1);
end
figure,plot(5:50,S);
