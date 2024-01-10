%% init

for init_project = 1 
    GH  = '/data_/mica1/03_projects/jessica/';
    
    % useful scripts
    addpath('/host/fladgate/local_raid/jessica/MICs/scripts/');
    addpath(genpath([GH, '/BrainSpace/matlab']));
    addpath(genpath([GH, 'plotSurfaceROIBoundary']));
    addpath([GH '/micasoft/matlab/useful'])
    addpath([GH '/useful']) 
    addpath('/data_/mica1/02_codes/NIfTI_20140122')
    addpath('/data_/mica1/02_codes/matlab_toolboxes/gifti-1.6/');
    fieldtripDir = '/data_/mica1/03_projects/jessica/fieldtrip';
    
    
    % directories
    datadir = '/host/oncilla/local_raid/jessica/microstructure_iEEG/data/';
    figdir = '/host/oncilla/local_raid/jessica/microstructure_iEEG/figures/july2023/';
    bids = '/data_/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/';
    ieeg_bids = '/host/oncilla/local_raid/iEEG-BIDS';

end


%% Patients
subjList = ['PX001'; 'PX005'; 'PX007'; 'PX009'; ...
    'PX010'; 'PX012'; 'PX015'; 'PX019'; 'PX023'; ...
    'PX028'; 'PX029'; 'PX031'; 'PX034'; 'PX040'; ...
    'PX041'; 'PX045'; 'PX050'; 'PX051'; 'PX053'; ...
    'PX065']; % omit PX004 because previous implant

ageAtMRI = [36, 26, 42, 25, 26, 33, 43, 31, ...
    40, 22, 18, 26, 51, 43, 43, 44, 32, 27, 41, 29];
sex = ['F'; 'F'; 'F'; 'F'; 'F'; 'F'; 'M'; ...
    'M'; 'F'; 'F'; 'F'; 'M'; 'M'; 'F'; 'F'; 'F'; ...
    'M'; 'F'; 'M'; 'M'];
mri_eeg_interval = [1, 5, 3, 4, 0, 47, 34, 13, ...
    0, 1, 2, 0, 3, 10, 0, 4, 8, 2, 0, 0];


%% Pre-proc each subject's iEEG data
    
resampling = 200;
load_data = 0;
for ii = 1:size(subjList,1)
    
    % Subject ID
    this_subj = subjList(ii,:);
    
    % Load list of contacts: list only contains sEEG contacts, scalp was
    % discarded
    filename = ['/host/oncilla/local_raid/iEEG-BIDS/sub-', this_subj,'/ieeg/annotation/keep_contacts.txt'];
    fileID = fopen(filename);
    C = textscan(fileID,'%s %s');
    contacts_load = {};
    for jj = 1:size(C{1},1)
        a1 = C{1,1}(jj);
        a2 = C{1,2}(jj);
        contacts_load{jj} = [a1{1}, ' ', a2{1}];
    end
    fclose(fileID);
    
    if load_data == 1
        
        disp(['loading... ', this_subj])
        
        cd('/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/');
        px_data{ii} = load(['sub-', this_subj,'_clean_ts.mat']);
        
    else
        
        disp(['beep boop time to compute: ', this_subj])
                
        % read edf header
        cd([ieeg_bids, '/sub-', this_subj, '/ieeg/']);
        cfg = [];
        tmp = dir('*.edf'); %edf has weird alphanumeric name
        edf_file = [ieeg_bids, '/sub-', this_subj, '/ieeg/', tmp.name];
        cfg.datafile = edf_file;
        cd([fieldtripDir, '/fileio'])
        hdr = ft_read_header(cfg.datafile);
        
        % Fitering and basic options
        cfg.continuous = 'yes';
        cfg.channel = contacts_load;
        cfg.bpfilter = 'yes';
        cfg.bpfreq = [0.5 80];
        cfg.bpfiltord = 1;
        cfg.bsfilter = 'yes';
        cfg.bsfreq = [59.5 60.5];
        
        % read edf file, with only specified channels
        cd(fieldtripDir)
        data = ft_preprocessing(cfg);
        
        % reorder channel list and timeseries to match contacts_load list
        tmpLabels = data.label;
        order = zeros(length(tmpLabels),1);
        for pos = 1:length(order)
            this_contact = string(tmpLabels{pos});
            idx = find(strcmpi(this_contact, contacts_load));
            order(pos) = idx;
        end
        tmpData = data.trial{1};
        dataReorder = tmpData(order,:);
        data.trial{1} = dataReorder;
        data.label = contacts_load';
        
        % bipolar re-referencing 
        cfg.reref = 'yes';
        cfg.refmethod = 'bipolar';
        cfg.refchannel = 'all';
        cfg.groupchans = 'yes';
        data2 = ft_preprocessing(cfg, data);
        clear data;
        
        % downsampling
        cfg.resamplefs = resampling;
        cfg.sampleindex = 'yes';
        dataDown = ft_resampledata(cfg,data2);
        idx = round(1:(data2.fsample/resampling):size(data2.trial{1},2));
        
        % Only keep good segments in timeseries
        segments = load([ieeg_bids, '/sub-', this_subj, '/ieeg/', 'sub-', this_subj, '_STSfile.mat']);
        items = segments.items;
        keep = zeros(1,size(data2.trial{1},2));
        for jj = 1:size(items,1)
            this_segment = [items(jj,2), items(jj,2)+items(jj,3)];
            keep(1,this_segment(1):this_segment(2)) = jj;
        end
        
        keepDown = keep(1,idx);
        itemsDown = items;
        for jj = 1:size(itemsDown,1)
            this_T = find(keepDown == jj);
            itemsDown(jj,2:3) = [this_T(1) this_T(end)-this_T(1)];
        end
        
        ts = dataDown.trial{1}(1:end-1,:);
        
        dataclean = {};
        for jj = 1:size(items,1)
            if jj == size(items,1)
                tmp = ts(:,itemsDown(jj,2):itemsDown(jj,2)+itemsDown(jj,3));
                dataclean{jj} = tmp - mean(tmp,2);
            else
                tmp = [ts(:,itemsDown(jj,2):itemsDown(jj,2)+itemsDown(jj,3)), zeros(size(ts,1),round(cfg.resamplefs)*2)]; % 2s padding between
                dataclean{jj} = tmp - mean(tmp,2);
            end
        end
        dataclean = horzcat(dataclean{:});
        dataLabels = dataDown.label(1:end-1,:);
        
        % Remove channels that bridge two different electrodes, or that
        % skip contacts e.g. RCP1-RCP9
        channel_keep = zeros(1,length(dataLabels));
        cd([ieeg_bids, '/sub-', this_subj, '/anat/nativepro/']);
        electrodeInfo = dir('electrode*.nii.gz');
        electrodeNames = {};
        for jj = 1:size(electrodeInfo,1)
            tmp = electrodeInfo(jj).name;
            first_split = strsplit(tmp,'_');
            second_split = strsplit(first_split{2},'.');
            electrodeNames{jj} = second_split{1};
            
            pat = ['\w*',lower(electrodeNames{jj}),'\w*'];
            for channel = 1:length(dataLabels)
                this_channel = dataLabels{channel};
                matchStr = regexp(lower(this_channel),pat,'match');
                if length(matchStr) == 2
                    
                    % Check if both contacts making up the channel are
                    % contiguous
                    matchStr1 = regexp(matchStr{1},'[0-9]\w*','match');
                    matchStr2 = regexp(matchStr{2},'[0-9]\w*','match');
                    
                    if str2num(matchStr2{1}) - str2num(matchStr1{1}) == 1
                        channel_keep(channel) = 1;
                    else
                        continue
                    end
                else
                    continue
                end
            end
        end
        dataclean = dataclean(logical(channel_keep),:);
        
        dataLabels2 = {};
        cpt=1;
        for aa = 1:length(channel_keep)
            if channel_keep(aa) == 1
                dataLabels2{cpt} = dataLabels{aa};
                cpt=cpt+1;
            else
            end
        end
        dataLabels = dataLabels2;
        
        cd('/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/');
        save(['sub-', this_subj,'_clean_ts.mat'], 'dataclean', 'dataLabels', 'contacts_load');
    end
end

%% Get PSD
f = 0.5:0.5:80;
fs = resampling;
log_transform = 1;
pxx_interp = {};
pxx_log = {};
load_data = 1;
for ii = 1:size(subjList,1)
    
    % Load data
    cd('/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni')
    
    this_subj = subjList(ii,:);
    disp(this_subj);
    
    if load_data==1
        disp('loading...')
        
        filename = ['/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/sub-', this_subj,'_pxx.csv'];
        pxx{ii} = csvread(filename);
        
        filename = ['/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/sub-', this_subj,'_pxxnorm.csv'];
        pxx_norm{ii} = csvread(filename);
        
        filename = ['/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/sub-', this_subj,'_pxxnorm_log.csv'];
        pxx_log{ii} = csvread(filename);
        
        filename = ['/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/sub-', this_subj,'_pxx_interp.csv'];
        pxx_interp{ii} = csvread(filename);
        
    else
        disp('beep boop time to compute')
        
        % compute psd and normalize
        px_data = load(['sub-',this_subj,'_clean_ts.mat']);
        ts = px_data.dataclean';
        pxx = [];
        pxx_norm = [];
        for channel = 1:size(ts,2)
            tmp = ts(:,channel);
            tmp(tmp == 0) = [];
            pxx(:,channel) = pwelch(tmp,2*fs,1*fs,f,fs);
            pxx_norm(:,channel) = pxx(:,channel)/sum(pxx(:,channel));
        end
        
        filename = ['/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/sub-', this_subj,'_pxx.csv'];
        csvwrite(filename, pxx);
        filename = ['/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/sub-', this_subj,'_pxxnorm.csv'];
        csvwrite(filename, pxx_norm);
        if log_transform == 1
            pxx_norm_log = log(pxx_norm);
            filename = ['/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/sub-', this_subj,'_pxxnorm_log.csv'];
            csvwrite(filename, pxx_norm_log);
            
            for channel = 1:size(ts,2)
                % interp
                idx1 = find(f == 59);
                idx2 = find(f == 61);
                vq = interp1([f(idx1) f(idx2)],[pxx_norm_log(idx1,channel) pxx_norm_log(idx2,channel)],f(idx1):0.5:f(idx2));
                pxx_interp{ii}(:,channel) = pxx_norm_log(:,channel);
                pxx_interp{ii}(idx1:idx2,channel) = vq;
            end
            filename = ['/host/oncilla/local_raid/jessica/microstructure_iEEG/data/ieeg-mni/sub-', this_subj,'_pxx_interp.csv'];
            csvwrite(filename, pxx_interp{ii});
        end
        
        pxx_log{ii} = pxx_norm_log;
        
    end
    
    
end


