clear; close all;
fdt = readtable('C:\Users\alex\OneDrive - California Institute of Technology\Documents\GitHub\Tentin-Quarantino\data\us\covid\nyt_us_counties.csv');
% Fix NYC stuff
fdt(strcmp(fdt.county, 'New York City'),'fips') = {36061};

fdt = remove_nan_fips(fdt);
[fdt, fips_to_max_deaths] = add_max_deaths(fdt);

% fips_to_max_deaths = create_fips_to_max_deaths(fdt);

disp(fips_to_max_deaths(1:100,:))

% fdt_sorted = sortrows(fdt,'max_deaths');

county = 'hennepin'; state = 'minnesota';
plot_fips(fdt,{county,state});
pause()

list_of_fips = fips_to_max_deaths.fips;
for fips = list_of_fips'
    plot_fips(fdt,fips);
    pause()
end