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
        _log.info(f'Found {self.n_recs} recordings\n\t:'+'\n\t'.join([x.name for x in self.ni_fns]))
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
        alf_obj_out = alfio.load_object(self.alf_path,object_name,extra=f't0',short_keys=True)
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
        self.alf_objects = object_names

    def concatenate_all_objects(self,save = True,overwrite=False):
        for object_name in self.alf_objects:
            obj_cat = self.concatenate_triggers(object_name)
            #TODO: Save
            #TODO: Overwrite
        



    

            
            
