function st = select_county(full_data_table,fips_selection)
if isnumeric(fips_selection)
    fips = fips_selection;
    if length(fips) == 1
        st = full_data_table(full_data_table.fips == fips,:);
    else
        disp_err('select_county() : FIPS LENGTH WAS NOT 1, VECTOR OF FIPS NOT SUPPORTED')
        return
    end
else
    county = fips_selection{1};
    state = fips_selection{2};
    st = full_data_table(strcmpi(full_data_table.county,county),:);
end
if size(st,1) == 0
    disp_err({['ERROR: select_county(~,' num2str(fips_or_county_name) ') could not find this county'],'Input arguments were: ',fips_selection})
    
    return
end
end
