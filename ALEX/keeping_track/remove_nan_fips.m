function fdt = remove_nan_fips(fdt)
idxs = find(isnan(fdt.fips));
fdt(idxs,:) = [];
end