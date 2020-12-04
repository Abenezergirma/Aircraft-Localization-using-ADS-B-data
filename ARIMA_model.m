%%%%%%% Abenezer Taye %%%%%%%%%%%%%
%%%%%%% Fall 2020 %%%%%%%%%%%%%%%%
%%%%%%% ARIMA model Implementaion of aircraft localization %%%%%%%%%%%
T = readtable('DataFrame.csv');
time = T.time;

latitude = T.lat;
longitude = T.lon;
velocity = T.velocity;
altitude = T.geoaltitude;

t = [];
for i=1:length(time)
t(i,:) = etime(datevec(time(i)),datevec(time(1)));
end

%% Train test split

n = numel(latitude);
ns = floor(n/2)+3150;
Ts = 10;

time_train = t(1:ns,:);
time_test = t((ns+1:end),:);

latitude_train = latitude(1:ns,:);
latitude_test = latitude((ns+1:end),:);

longitude_train = longitude(1:ns,:);
longitude_test = longitude((ns+1:end),:);

altitude_train = altitude(1:ns,:);
altitude_test = altitude((ns+1:end),:);


%% Plots of the data

subplot(2,3,1)
plot(t,latitude, 'LineWidth', 1.2)
grid on
title('Latitude Profile')
xlabel('time(sec)')
ylabel('latitude')

subplot(2,3,2)
plot(t,longitude, 'LineWidth', 1.2)
grid on
title('Longitude Profile')
xlabel('time(sec)')
ylabel('longitude')

subplot(2,3,3)
plot(t,altitude, 'LineWidth', 1.2)
grid on
title('Altitude Profile')
xlabel('time(sec)')
ylabel('altitude')

subplot(2,3,4)
plot(time_train,latitude_train,'LineWidth', 1.2)
hold on
plot(time_test,latitude_test,'LineWidth', 1.2)
legend('latitude train','latitude test');
grid on
title('Latitude Profile')
xlabel('time(sec)')
ylabel('latitude')

subplot(2,3,5)
plot(time_train,longitude_train,'LineWidth', 1.2)
hold on
plot(time_test,longitude_test,'LineWidth', 1.2)
legend('longitude train','longitude test');
grid on
title('Longitude Profile')
xlabel('time(sec)')
ylabel('longitude')

subplot(2,3,6)
plot(time_train,altitude_train,'LineWidth', 1.2)
hold on
plot(time_test,altitude_test,'LineWidth', 1.2)
legend('altitude train','altitude test');
grid on
title('Altitude Profile')
xlabel('time(sec)')
ylabel('altitude')



%% Latitude prediction
[latitude_forecast,Lat_MSE] = forecast(ARIMA_latitude_train1,50,'Y0',latitude_train1);

UB = latitude_forecast + 1.96*sqrt(Lat_MSE);
LB = latitude_forecast - 1.96*sqrt(Lat_MSE);



timeF = time_test(1:50);

time_Tr = [time_train(end-20:end) ;time_test(1:70)];  
time_Tst = [time_train(end-20:end) ;time_test(1:70)];  

latitude_Tr = [latitude_train(end-20:end); latitude_test(1:70)];  
latitude_Tst = [latitude_train(end-20:end); latitude_test(1:70)]; 

plot(time_Tr,latitude_Tr,'LineWidth', 1.2)
hold on
plot(timeF,latitude_forecast,'r','LineWidth',2)
plot(timeF,UB,'k--','LineWidth',1.5)
plot(timeF,LB,'k--','LineWidth',1.5)
grid on
xlabel('time')
ylabel('latitude')
legend('actual trajectory','predicted trajectory', 'prediction interval')
title('Latitude Prediction')

%summarize(ARIMA_latitude_train1)
%% Longitude prediction
[longitude_forecast,Lon_MSE] = forecast(ARIMA_longitude_train1,50,'Y0',longitude_train1);

UB = longitude_forecast + 1.96*sqrt(Lon_MSE);
LB = longitude_forecast - 1.96*sqrt(Lon_MSE);



timeF = time_test(1:50);

time_Tr = [time_train(end-20:end) ;time_test(1:70)];  
time_Tst = [time_train(end-20:end) ;time_test(1:70)];  

longitude_Tr = [longitude_train(end-20:end); longitude_test(1:70)];  
longitude_Tst = [longitude_train(end-20:end); longitude_test(1:70)]; 

plot(time_Tr,longitude_Tr,'LineWidth', 1.2)
hold on
plot(timeF,longitude_forecast,'r','LineWidth',2)
plot(timeF,UB,'k--','LineWidth',1.5)
plot(timeF,LB,'k--','LineWidth',1.5)
grid on
xlabel('time')
ylabel('longitude')
legend('actual trajectory','predicted trajectory', 'prediction interval')
title('Longitude Prediction')
%summarize(ARIMA_longitude_train1)
%% Altitude Prediction 
[altitude_forecast,Alt_MSE] = forecast(ARIMA_altitude_train1,100,'Y0',altitude_train1);

UB = altitude_forecast + 1.96*sqrt(Alt_MSE);
LB = altitude_forecast - 1.96*sqrt(Alt_MSE);



timeF = time_test(1:100);

time_Tr = [time_train(end-20:end) ;time_test(1:120)];  
time_Tst = [time_train(end-20:end) ;time_test(1:120)];  

altitude_Tr = [altitude_train(end-20:end); altitude_test(1:120)];  
altitude_Tst = [altitude_train(end-20:end); altitude_test(1:120)]; 

plot(time_Tr,altitude_Tr,'LineWidth', 1.2)
hold on
plot(timeF,altitude_forecast,'r','LineWidth',2)
plot(timeF,UB,'k--','LineWidth',1.5)
plot(timeF,LB,'k--','LineWidth',1.5)
grid on
xlabel('time')
ylabel('altitude')
legend('actual trajectory','predicted trajectory', 'prediction interval')
title('altitude Prediction')
%summarize(ARIMA_altitude_train1)