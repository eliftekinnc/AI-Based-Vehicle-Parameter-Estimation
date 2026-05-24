clear; clc;

num_simulations = 1000; % Üretilecek senaryo sayısı
dataset = struct();
t_end = 10; % Her sürüş 10 saniye
dt = 0.1;   % 100 örnek 
t_span = 0:dt:t_end;

fprintf('Veri üretimi başladı...\n');

for i = 1:num_simulations
    % Rastgele Parametreler
    m = 1200 + rand()*800;      % 1200 - 2000 kg
    Iz = 2000 + rand()*1000;    % Atalet momenti
    Cf = 15000 + rand()*10000;  % Ön lastik sertliği
    Cr = 15000 + rand()*10000;  % Arka lastik sertliği
    lf = 1.2; lr = 1.6;         % Aks mesafeleri 
    
    params = [m, Iz, Cf, Cr]; % Hedef etiketimiz
    
    % Rastgele Sürüş Girişleri (u)
    % Direksiyon: Rastgele zamanlarda değişen sinüs dalgaları
    delta = 0.1 * sin(2*pi*rand()*t_span) .* exp(-0.1*t_span);

    % Boylamsal Kuvvet: Gaz/Fren
    Fx = 500 + 1000 * rand(size(t_span)); 

    % Diferansiyel Denklemleri Çöz 
    % Durumlar: [vx, vy, r]

    x0 = [10; 0; 0]; % Başlangıç hızı 10 m/s
    
    % Basit bir ileri Euler veya ODE çözümü
    vx = zeros(size(t_span)); vy = zeros(size(t_span)); r = zeros(size(t_span));
    vx(1) = x0(1); vy(1) = x0(2); r(1) = x0(3);
    
    for k = 1:length(t_span)-1
        % Slip angles (Kayma açıları)
        alpha_f = delta(k) - (vy(k) + lf*r(k))/vx(k);
        alpha_r = -(vy(k) - lr*r(k))/vx(k);
        
        % Yanal kuvvetler
        Fyf = Cf * alpha_f;
        Fyr = Cr * alpha_r;
        
        % Hareket denklemleri
        dvx = (Fx(k) - Fyf*sin(delta(k)) + m*vy(k)*r(k)) / m;
        dvy = (Fyf*cos(delta(k)) + Fyr - m*vx(k)*r(k)) / m;
        dr = (lf*Fyf*cos(delta(k)) - lr*Fyr) / Iz;
        
        % Güncelleme
        vx(k+1) = vx(k) + dvx * dt;
        vy(k+1) = vy(k) + dvy * dt;
        r(k+1) = r(k) + dr * dt;
    end
    
    dataset(i).inputs = [delta', Fx'];           % Girişler (Steer, Force)
    dataset(i).outputs = [vy', r'];             % Ölçümler (Lat_vel, Yaw_rate)
    dataset(i).true_states = [vx', vy', r'];    % Gerçek stateler
    dataset(i).params = params;                
end

save('otonom_arac_dataset.mat', 'dataset', '-v7.3');
fprintf('1000 adet sürüş verisi "otonom_arac_dataset.mat" olarak kaydedildi.\n');