function fips_to_max_deaths = create_fips_to_max_deaths(fdt)
unique_fdt = unique(fdt,'fips');
fips_to_max_deaths = 
    