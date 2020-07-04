function plot_fips(full_data_table,fips_selection)

st = select_county(full_data_table,fips_selection); % full_data_table(full_data_table.fips == fips,:);
fips = st.fips(1);

f1 = figure2();
p1 = plot(st.date,st.deaths,'k.');
set_axis_properties(st);
xlabel('date');
ylabel('# deaths total');
title([st.county{1} ', ' st.state{1} ' (' num2str(fips) ')']);
legend('NYT data','location','northwest')

deaths_per_day = [0; diff(st.deaths)];
smoothing_time_scale = 3;
smooth_deaths_per_day = imgaussfilt(deaths_per_day,smoothing_time_scale);


f2 = figure2();
p2 = plot(st.date,deaths_per_day,'r.');
hold on
p2 = plot(st.date,smooth_deaths_per_day,'r-');
set_axis_properties(st);
xlabel('date');
ylabel('# deaths per day');
title([st.county{1} ', ' st.state{1} ' (' num2str(fips) ')']);
legend('raw NYT data','smoothed NYT data','location','northwest')

infected_per_day = [0; diff(st.cases)];
smoothing_time_scale = 3;
smooth_infected_per_day = imgaussfilt(infected_per_day,smoothing_time_scale);

f2 = figure2();
p2 = plot(st.date,infected_per_day,'b.');
hold on
p2 = plot(st.date,smooth_infected_per_day,'b-');
set_axis_properties(st);
xlabel('date');
ylabel('# infected per day');
title([st.county{1} ', ' st.state{1} ' (' num2str(fips) ')']);
legend('raw NYT data','smoothed NYT data','location','northwest')
end

function set_axis_properties(st)
fips = st.fips(1);
downsample_number = 3;
a1 = gca();
a1.YAxis.Exponent = 0;
ytickformat('%.0f');
% a1.XRuler.TickLabelFormat = 'MMM W';
a1.XRuler.TickLabelFormat = 'MM/dd';
xticks(downsample(st.date,downsample_number));
axis tight;
xtickangle(80);
grid minor
grid on
a1.YRuler.MinorTick = 'on';
a1.XRuler.MinorTick = 'on';
a1.XRuler.MinorTickValues = st.date;
end