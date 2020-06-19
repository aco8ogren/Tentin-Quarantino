function [fdt,fips_to_max_deaths] = add_max_deaths(fdt)
list_of_fips = unique(fdt.fips);
fdt.max_deaths = nan(size(fdt,1),1);
% fips_to_max_deaths = table({'county'},{'state'},{'fips'},{'max_deaths'});
tic
fips_for_table = [];
state = [];
county = [];
max_deaths_for_table = [];
for fips = list_of_fips'
    st = select_county(fdt,fips);
    max_deaths = max(st.deaths);
    idxs = find(fdt.fips == fips);
    fdt(idxs,'max_deaths') = table(max_deaths*ones(length(idxs),1));
    fips_for_table = [fips_for_table fips];
    state = [state st.state(1)];
    county = [county st.county(1)];
    max_deaths_for_table = [max_deaths_for_table max_deaths];
end
max_deaths = max_deaths_for_table;
fips = fips_for_table;
fips_to_max_deaths = table(fips', county', state', max_deaths');
fips_to_max_deaths.Properties.VariableNames = {'fips','county','state','max_deaths'};
fips_to_max_deaths = sortrows(fips_to_max_deaths,'max_deaths','descend');
toc
end