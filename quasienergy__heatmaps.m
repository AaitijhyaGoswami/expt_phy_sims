function dynamic_local_2()
% Simulation of 1D Tight-Binding Model with Driving

    %% SYSTEM CONFIG
    % Physical Parameters
    N = 51;             % Number of sites
    J = 1;              % Tunneling amplitude
    center_site = 26;   % Initial localization site
    
    % Driving Parameters
    omega = 10;         % Driving frequency (High frequency regime)
    ratios = [0, 1, 2.4048, 3]; % K/omega ratios
    
    % Simulation Parameters
    T_total = 40;       % Total simulation time
    dt = 0.1;           % Sampling time step
    time_axis = 0:dt:T_total;
    
    % Solver Options
    opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);

    %% TIME EVOL AND PLOTTING
    % Using a tiled layout for a clean, single-window view of all propagations
    figure('Name', 'Time Evolution of Probability Density', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for i = 1:length(ratios)
        ratio = ratios(i);
        K = ratio * omega;
        
        % Initial State: Localized at center
        psi0 = zeros(N,1);
        psi0(center_site) = 1;
        
        % Defining Hamiltonian function handle
        H_func = @(t, psi) hamiltonian_1d(t, psi, N, J, K, omega);
        
        % Solving Time-Dependent Schrödinger Equation
        [~, PSI_sol] = ode45(H_func, time_axis, psi0, opts);
        
        % Calculatinh |psi|^2
        Prob_Density = abs(PSI_sol).^2;
        
        % Plotting Current Tile
        nexttile;
        imagesc(time_axis, 1:N, Prob_Density');
        set(gca, 'YDir', 'normal', 'FontSize', 12, 'LineWidth', 1.5);
        colormap(jet);
        
        % Titles and formatting
        if ratio == 0
            title_str = "Static (Ballistic)";
            subtitle_str = "J_{eff} = J";
        elseif abs(ratio - 2.4048) < 0.01
            title_str = "Dynamic Localization";
            subtitle_str = "K/\omega \approx 2.404, J_{eff} \approx 0";
        else
            title_str = sprintf("Driven: K/\\omega = %.1f", ratio);
            eff_J = J * besselj(0, ratio);
            subtitle_str = sprintf("J_{eff} \\approx %.2f J", eff_J);
        end
        
        title({title_str, subtitle_str}, 'FontWeight', 'bold');
        xlabel('Time (1/J)');
        ylabel('Site Index');
        clim([0 0.4]); % Standardizing color limit for comparison
    end
    
    % Global Colorbar
    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Label.String = '|\psi|^2 Probability';
    cb.Label.FontSize = 12;

    %% FLOQUET QUASIENERGY SPECTRUM
    % Calculating evolution operator over one period T = 2*pi/omega
    disp('Calculating Floquet Spectrum (this may take a moment)...');
    
    T_period = 2*pi/omega;
    figure('Name', 'Floquet Quasienergy Spectrum', 'Color', 'w', 'Position', [150, 150, 1200, 400]);
    t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, 'Floquet Quasienergy Spectrum', 'FontSize', 16, 'FontWeight', 'bold');

    % Looping only through the driven cases (indices 2, 3, 4)
    driven_indices = 2:4; 
    
    for i = driven_indices
        ratio = ratios(i);
        K = ratio * omega;
        
        % Constructing Floquet Operator U(T) column by column
        U_Floquet = zeros(N, N);
        
        for n = 1:N
            basis_vec = zeros(N,1);
            basis_vec(n) = 1;
            [~, sol] = ode45(@(t,p) hamiltonian_1d(t,p,N,J,K,omega), [0 T_period], basis_vec, opts);
            U_Floquet(:, n) = sol(end, :).';
        end
        
        % Diagonalizing U to get eigenvalues lambda
        eigenvals_U = eig(U_Floquet);
        
        % Quasienergies: epsilon = i/T * log(lambda)
        % Taking the real part (imaginary part should be 0 for unitary U)
        quasienergies = real(1i/T_period * log(eigenvals_U));
        quasienergies = sort(quasienergies);
        
        % Theoretical Bandwidth (4 * J_eff)
        eff_J = abs(J * besselj(0, ratio));
        bandwidth_theory = 4 * eff_J;
        
        % Plotting Spectrum
        nexttile;
        plot(quasienergies, 'o-', 'LineWidth', 1.5, 'MarkerSize', 4, 'Color', [0 0.4470 0.7410]);
        grid on; box on;
        set(gca, 'FontSize', 12, 'LineWidth', 1.2);
        
        xlim([0 N+1]);
        ylim([-2.5 2.5]); % Fixed y-limits for easy visual comparison
        
        xlabel('State Index n');
        if i == 2; ylabel('Quasienergy \epsilon'); end
        
        title(sprintf('K/\\omega = %.3f', ratio), 'FontWeight', 'bold');
        subtitle(sprintf('Bandwidth \\approx %.2f (Theory: %.2f)', ...
            max(quasienergies)-min(quasienergies), bandwidth_theory));
    end
end

%% HELPER FUNCTIONS DEFINITION

function dpsidt = hamiltonian_1d(t, psi, N, J, K, omega)
    % Computes time derivative dpsi/dt = -i*H(t)*psi
    
    % Potential Energy Term (Diagonal)
    % V(n) = K * cos(omega*t) * n
    % Centering the potential at (N+1)/2 to keep the lattice symmetric
    n_indices = (1:N)';
    V_site = K * cos(omega * t) * (n_indices - (N+1)/2); 
    
    % Kinetic Energy Term (Off-diagonal / Hopping)
    % H_hopping = -J (|n><n+1| + |n+1><n|)
    % Implementing via vectorized shifting for speed
    psi_left = [0; psi(1:end-1)];  % Shifts indices n -> n+1
    psi_right = [psi(2:end); 0];   % Shifts indices n -> n-1
    
    H_psi_kinetic = -J * (psi_left + psi_right);
    H_psi_potential = V_site .* psi;
    
    % 3. Applying total H to psi
    H_psi = H_psi_kinetic + H_psi_potential;
    
    % Schrödinger Equation: dpsi/dt = -i * H * psi
    dpsidt = -1i * H_psi;
end
