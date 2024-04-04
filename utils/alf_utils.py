import one.alf.io as alfio
from pathlib import Path
import numpy as np
import spikeglx
import logging
logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


class Recording:
    def __init__(self,session_path):
        self.session_path = session_path
        self.raw_ephys_path = self.session_path.joinpath('raw_ephys_data')
        assert self.raw_ephys_path.is_dir(),'Raw ephys data folder does not exist as expected'
        self.alf_path = self.session_path.joinpath('alf')
        assert self.raw_ephys_path.is_dir(),'Alf path does not exist as expected'
        self.ni_fns = list(self.raw_ephys_path.glob('*nidq.bin'))
        self.n_recs = len(self.ni_fns)
        _log.info(f'Found {self.n_recs} recordings:\n\t'+'\n\t'.join([x.name for x in self.ni_fns]))
        self.get_breaks_times()
        self.list_all_alf_objects()


    def get_breaks_times(self):
        # I think I am repeating myself here
        breaks = [0]
        for ni_fn in self.ni_fns:
            SR = spikeglx.Reader(ni_fn)
            breaks.append(breaks[-1]+SR.meta['fileTimeSecs'])
        self.breaks = np.array(breaks)
    

    def concatenate_triggers(self,object_name):
        try:
            alf_obj_out = alfio.load_object(self.alf_path,object_name,extra=f't0',short_keys=True)
        except:
            # Doing it this way so that if the files can be loaded, we do not error, but do throw an error if the problem is not the trigger label
            alf_obj_out = alfio.load_object(self.alf_path,object_name,short_keys=True)
            _log.info('Triggers not found. Has this already been concatenated?')
            return(alf_obj_out)

        for ii in range(1,self.n_recs):
            alf_obj= alfio.load_object(self.alf_path,object_name,extra=f't{ii:0.0f}',short_keys=True)
            for k in alf_obj.keys():
                if len(alf_obj[k])==0:
                    continue
                if k=='intervals' or k=='times':
                    alf_obj[k] +=self.breaks[ii]
                alf_obj_out[k] = np.concatenate([alf_obj_out[k],alf_obj[k]])
        return(alf_obj_out)
    

    def list_all_alf_objects(self):
        object_parts = alfio.filter_by(self.alf_path)[1]
        object_names = list(set([x[1] for x in object_parts]))
        object_names.sort()
        self.alf_objects = object_names


    def concatenate_all_objects(self,save = True,overwrite=True):

        for object_name in self.alf_objects:
            _log.info(f'Concatenating {object_name}')
            obj_cat = self.concatenate_triggers(object_name)
            old_files,old_file_parts = alfio.filter_by(self.alf_path,object=object_name)
            if old_file_parts[0][-2]==None:
                _log.info(f'Skipping concatenation of {object_name}. It appears to be concatenated')
                continue

            extensions = list(set([x[-1] for x in old_file_parts]))
            namespaces = list(set([x[0] for x in old_file_parts]))
            attributes = list(set([x[2] for x in old_file_parts]))
            assert len(namespaces)==1,f'Multiple namespaces found for {object_name}:{namespaces}'
            namespace = namespaces[0]

            if save:
                if 'table' in attributes:
                    table_fn = alfio.files.spec.to_alf(object_name,'table',extension='pqt',namespace=namespace)
                    _log.info(f'Saving concatenated {object_name} to parquet')
                    obj_cat.to_df().to_parquet(self.alf_path.joinpath(table_fn))
                else:
                    assert(len(extensions)==1),'There should only be one extension'
                    assert(extensions[0]=='npy'),'There should only be one extension'
                    _log.info(f'Saving concatenated {object_name} to npy')
                    alfio.save_object_npy(self.alf_path,obj_cat,object_name,namespace=namespace)
            if overwrite:
                _log.info('Removing old files')
                _log.debug(f'\n\t'+'\n\t'.join(old_files))
                for fn in old_files:
                    self.alf_path.joinpath(fn).unlink()
            
        
    def load_spikes():
        #TODO: Only keep good cells?
        #TODO: Work with multiple probes?
        pass