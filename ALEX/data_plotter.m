clear; close all;
t = readtable('C:\Users\alex\OneDrive - California Institute of Technology\Documents\GitHub\Tentin-Quarantino\data\us\covid\nyt_us_counties.csv');
% Fix NYC stuff
t(strcmp(t.county, 'New York City'),'fips') = {36061};
fips = 36061;
tt = t(t.fips == fips,:);
figure2()
plot(tt.date,tt.deaths);
% datetick('x','mm/dd');
% xticks(downsample(tt.date,1));
xtickangle(45);