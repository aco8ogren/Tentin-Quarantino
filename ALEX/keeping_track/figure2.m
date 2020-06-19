% [EDITED, 2018-06-05, typos fixed]
function FigHandle = figure2(varargin)
% Catch creation of figure with disabled visibility:
indexVisible = find(strncmpi(varargin(1:2:end), 'Vis', 3));
if ~isempty(indexVisible)
    paramVisible = varargin(indexVisible(end) + 1);
else
    paramVisible = get(0, 'DefaultFigureVisible');
end
MP = get(0, 'MonitorPositions');
if size(MP, 1) == 1  % Single monitor
    Shift = MP(1, 1:2) - [0 50]; % I subtract [0 50] because I'm using the task bar on top ...
    FigH = figure(varargin{:},'Visible','off');
    set(FigH, 'Units', 'pixels');
    pos      = get(FigH, 'Position');
    set(FigH, 'Position', [pos(1:2) + Shift, pos(3:4)], ...
        'Visible', paramVisible);
else                 % Multiple monitors

    Shift    = MP(2, 1:2) - [0 50]; % I subtract [0 50] because I'm using the task bar on top ...
    FigH     = figure(varargin{:}, 'Visible', 'off');
    set(FigH, 'Units', 'pixels');
    pos      = get(FigH, 'Position');
    set(FigH, 'Position', [pos(1:2) + Shift, pos(3:4)], ...
        'Visible', paramVisible);
end
if nargout ~= 0
    FigHandle = FigH;
end