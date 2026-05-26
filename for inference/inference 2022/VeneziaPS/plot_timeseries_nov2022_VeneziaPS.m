%%

clear, clc

%% Load 

% Observations - Punta della Salute
addpath('C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\surge_ml\tests\inference 2022\observations')
sta = load('meteoDTNLP13h_TG_VeneziaPS_2022.txt');

% Observations - CNR platform
addpath('C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\surge_ml\tests\inference 2022\observations')
sta2 = load('meteoDTNLP13h_TG_CNR_2022.txt');

% Copernicus forecast
addpath('C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\surge_ml\tests\inference 2022\medphysics')
cop = load('meteoDTNLP13h_cmemsforecast_CNR_2022-2023.txt');

% ML model 1 - MLR
addpath('C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\surge_ml\tests\inference 2022\VeneziaPS\MLR-MADc')
shy = load('mlrModel_Run2_2022_inference_RAW.txt');

% ML model 2 - LSTMh
addpath('C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\surge_ml\tests\inference 2022\VeneziaPS\LSTMh-MADc')
shy2 = load('lstmHybridModel_Run16_2022_inference_RAW.txt');

% ML model 2 - MLP
% addpath('C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\Experiments_v6_v2\surge-ml_v2 - sliding-window\downscaling 2022\VeneziaPS\MLP-MADc')
% shy2 = load('mlpModelLP13h_VeneziaPS_2022_checkpoint33_v2026.txt');

%%

% Observations 
x = sta(:,8)-mean(sta(:,8),'omitnan');
x2 = sta2(:,8)-mean(sta2(:,8),'omitnan');

% Copernicus forecast
f = cop(:,8)-mean(cop(:,8));

% ML models
y1 = shy(:,8)-mean(shy(:,8));
y2 = shy2(:,8)-mean(shy2(:,8));

%% Plot complete downscaling period

figure(1)
set(gcf,'position',[60 60 1700 500])

% Observations
plot(sta(:,7),x,'k','LineWidth',1.4)
hold on
plot(sta2(:,7),x2,'r','LineWidth',1.4)
hold on

% Copernicus forecast
plot(cop(:,7),f,'m','LineWidth',1.4)
hold on

% ML models
plot(shy(:,7),y1,'color',[0 0.4470 0.7410],'LineWidth',1.2)
hold on
plot(shy2(:,7),y2,'color',[0.8500 0.3250 0.0980],'LineWidth',1.2)

% Apply Century Gothic font to all text elements
set(gca, 'FontName', 'Century Gothic');  % Axes labels and ticks
title('Punta della Salute, 2022', 'FontName', 'Century Gothic');
xlabel('Time [mm/yy]', 'FontName', 'Century Gothic');
ylabel('Surge [m]', 'FontName', 'Century Gothic');

datetick('x','mm/yy','keepticks')
grid on
ax = gca;
ax.FontSize = 16;
%ax.TitleFontSizeMultiplier = 1.1;
ax.TitleFontWeight = 'bold';

legend({'Tide gauge: Punta della Salute (PS)','Tide gauge: CNR platform',...
    'Copernicus forecast CNR',...
    'MLR-MADc^2 (trained on PS)', 'LSTMh-MADc^2 (trained on PS)'},...
    'Location', 'east', 'NumColumns', 1, 'FontSize', 13);
lgd = legend('Location', 'eastoutside', 'Orientation', 'horizontal');
lgd.Title.String = 'Legend';
lgd.FontName = 'Century Gothic';

% Export on high resolution
addpath('C:\Users\Rodrigo\Desktop\Rodrigo\Matlab functions\altmany-export_fig-3.40.0.0')
set(gcf, 'Color', 'w');
%export_fig('Timeseries_2022_traintestTG_VeneziaPS_MLR_LSTMh_HQ.png','-m2.5')

%% Specific storm surge event 

yearselec = 2022;
monthselec = 11;

figure(2)
set(gcf,'position',[60 60 1800 650])

ai_1 = datenum(yearselec,monthselec,1,00,00,00):datenum(0000,0,0,1,00,00):datenum(yearselec,monthselec,31,23,00,00);
ix_plot1 = find(shy(:,1)==yearselec(1) & shy(:,2)==monthselec(1));

% Observations
plot(sta(:,7),x,'k','LineWidth',1.8)
hold on
plot(sta2(:,7),x2,'r','LineWidth',1.8)
hold on

% Copernicus forecast
plot(cop(:,7),f,'m','LineWidth',1.4)
hold on

% ML models
plot(shy(:,7),y1,'color',[0 0.4470 0.7410],'LineWidth',1.6)
hold on
plot(shy2(:,7),y2,'color',[0.8500 0.3250 0.0980],'LineWidth',1.6)

grid on

% Apply Century Gothic font to all text elements
set(gca, 'FontName', 'Century Gothic');  % Axes labels and ticks
title('Punta della Salute (November, 2022)', 'FontName', 'Century Gothic');
xlabel('Time [dd/mm/yy]', 'FontName', 'Century Gothic');
ylabel('Surge [m]', 'FontName', 'Century Gothic');

xlim([ai_1(1) ai_1(end)])
ylim([-0.3 1.1])
ax = gca;
ax.XTick = shy(ix_plot1(1):60:ix_plot1(end),7);
datetick('x','dd/mm/yy','keepticks')
ax.FontSize = 20;
%ax.TitleFontSizeMultiplier = 1.1;
ax.TitleFontWeight = 'bold';

legend({'TG: Punta della Salute (PS)','TG: CNR platform',...
    'Med-Physics',...
    'MLR-MADc^2', 'LSTMh-MADc^2'},...
    'Location', 'east', 'NumColumns', 5, 'FontSize', 18);
lgd = legend('Location', 'southoutside', 'Orientation', 'horizontal');
lgd.Title.String = 'Legend';
lgd.FontName = 'Century Gothic';

% Export on high resolution
addpath('C:\Users\Rodrigo\Desktop\Rodrigo\Matlab functions\altmany-export_fig-3.40.0.0')
set(gcf, 'Color', 'w');
%export_fig('Timeseries_november2022_VeneziaPS_MLR_LSTMh.png','-m2.5')
%export_fig('Timeseries_november2022_VeneziaPS_MLR_MLP.png','-m2.5')

%%