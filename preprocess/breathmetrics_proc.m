function breathmetrics_proc(p_load,trigger_label)
%% given a directory abd a trigger label (e.g. 't0'), flexibly loads the flow or pdiff data
% Computes the breathmetrics data.
% Aligns to diaphragm if it exists
% events, and append the breathmetrics results
verbose=1;
addpath(genpath('/active/ramirez_j/ramirezlab/nbush/helpers/npy-matlab'))
addpath(genpath('/active/ramirez_j/ramirezlab/nbush/helpers/breathmetrics'))

%% Load in the data
physiol_fn = [p_load '/' dir([p_load '/*physiology*' trigger_label '*pqt' ]).name];
physiol_times_fn = [p_load '/' dir([p_load '/*physiology*times*' trigger_label '*npy' ]).name];
breath_features_fn = [p_load '/' dir([p_load '/*breaths.table*' trigger_label '*pqt' ]).name];
breath_times_fn = [p_load '/' dir([p_load '/*breaths.times*' trigger_label '*npy' ]).name];

physiology = parquetread(physiol_fn);
physiology.t =readNPY(physiol_times_fn);
sr = 1/mean(diff(physiology.t));

has_dia = ismember('dia',physiology.Properties.VariableNames);
has_flow = ismember('flowmeter',physiology.Properties.VariableNames);
has_pdiff = ismember('pdiff',physiology.Properties.VariableNames);

% Only load in previously extracted diaphragm if dia was recorded
if has_dia
    disp('Has diaphragm')
    breaths = readtable(breath_features_fn,'FileType','text','Delimiter','\t');
end

% Overload to work on PDIFF or flowmeter signal
if has_flow & has_pdiff
    error('Flow and pressure differential signal are present. Not continuing');
end
if ~has_flow & ~has_pdiff
    error('No rodent airflow signal found');
end
if has_flow
    disp('Using flowmeter');
    flowmeter = physiology.flowmeter;
end
if has_pdiff
    disp('Using pdiff');
    pdiff = physiology.pdiff;
end

%% Do breathmetrics analysis on pleth
if has_flow
    flow2 = flowmeter/60/sr; %Convert from ml/min to ml/sample
    zScore=0;
    baselineCorrectionMethod = 'simple';
    bmObj = breathmetrics(flow2(:),sr,'rodentAirflow');
elseif has_pdiff
    zScore=1;
    baselineCorrectionMethod = 'sliding';
    bmObj = breathmetrics(pdiff(:),sr,'rodentAirflow');
end
bmObj.estimateAllFeatures(zScore, baselineCorrectionMethod, ...
    'simplify', verbose);
%% Register each diaphragm burst to an inhale
% Probably can be vecorized, but not worth my troubles now
if has_dia
    temporal_thresh = nanmean(breaths.postBI)*0.9;
    dia_pks = breaths.pk_time;
    inhale_pks = bmObj.inhalePeaks/sr;
    dpks = bsxfun(@minus,dia_pks,inhale_pks);
    [dpks,idx] = min(abs(dpks)');
    not_aligned = dpks>temporal_thresh;
    idx(not_aligned) = nan;

    %% Map to stats table
    dia_idx = find(isfinite(idx));
    bm_idx = idx(isfinite(idx));
    %% Map everything into the stats table
    breaths.inhale_peaks(dia_idx) = bmObj.inhalePeaks(bm_idx)/sr;
    breaths.exhale_troughs(dia_idx) = bmObj.exhaleTroughs(bm_idx)/sr;
    breaths.peak_inspiratory_flows(dia_idx) = bmObj.peakInspiratoryFlows(bm_idx);
    breaths.trough_expiratory_flows(dia_idx) = bmObj.troughExpiratoryFlows(bm_idx);
    breaths.inhale_onsets(dia_idx) = bmObj.inhaleOnsets(bm_idx)/sr;
    breaths.exhale_onsets(dia_idx) = bmObj.exhaleOnsets(bm_idx)/sr;
    breaths.inhale_offsets(dia_idx) = bmObj.inhaleOffsets(bm_idx)/sr;
    breaths.exhale_offsets(dia_idx) = bmObj.exhaleOffsets(bm_idx)/sr;
    breaths.inhale_time_to_peak(dia_idx) = bmObj.inhaleTimeToPeak(bm_idx)/sr;
    breaths.exhale_time_to_trough(dia_idx) = bmObj.exhaleTimeToTrough(bm_idx)/sr;
    breaths.inhale_volumes(dia_idx) = bmObj.inhaleVolumes(bm_idx);
    breaths.exhale_volumes(dia_idx) = bmObj.exhaleVolumes(bm_idx);
    breaths.exhale_durations(dia_idx) = bmObj.exhaleDurations(bm_idx)/sr;
    breaths.inhale_pause_onsets(dia_idx) = bmObj.inhalePauseOnsets(bm_idx)/sr;
    breaths.exhale_pause_onsets(dia_idx) = bmObj.exhalePauseOnsets(bm_idx)/sr;
    breaths.inhale_pause_durations(dia_idx) = bmObj.exhalePauseDurations(bm_idx)/sr;
    breaths = standardizeMissing(breaths,0);


else
    breaths=table;
    breaths.inhale_peaks = bmObj.inhalePeaks(:)/sr;
    breaths.exhale_troughs = bmObj.exhaleTroughs(:)/sr;
    breaths.peak_inspiratory_flows = bmObj.peakInspiratoryFlows(:);
    breaths.trough_expiratory_flows = bmObj.troughExpiratoryFlows(:);
    breaths.inhale_onsets = bmObj.inhaleOnsets(:)/sr;
    breaths.exhale_onsets = bmObj.exhaleOnsets(:)/sr;
    breaths.inhale_offsets = bmObj.inhaleOffsets(:)/sr;
    breaths.exhale_offsets = bmObj.exhaleOffsets(:)/sr;
    breaths.inhale_time_to_peak = bmObj.inhaleTimeToPeak(:)/sr;
    breaths.exhale_time_to_trough = bmObj.exhaleTimeToTrough(:)/sr;
    breaths.inhale_volumes = bmObj.inhaleVolumes(:);
    breaths.exhale_volumes = bmObj.exhaleVolumes(:);
    breaths.exhale_durations = bmObj.exhaleDurations(:)/sr;
    breaths.inhale_pause_onsets = bmObj.inhalePauseOnsets(:)/sr;
    breaths.exhale_pause_onsets = bmObj.exhalePauseOnsets(:)/sr;
    breaths.inhale_pause_durations = bmObj.exhalePauseDurations(:)/sr;
    breaths = standardizeMissing(breaths,0);

end
%% write modified stats table
parquetwrite(breath_features_fn,breaths);
writeNPY(breaths.inhale_onsets,breath_times_fn)
exit;
