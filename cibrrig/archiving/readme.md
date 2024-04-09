# Archiving

These scripts operates on a "run" folder, which is the parent directory that Spikeglx writes to. Runnames should be the mouse ID.

```
sglx_data/
├─ Subjects/
│  ├─ runname/ **<- Run Folder
│  │  ├─ runname_g0/
│  │  ├─ runname_g1/
```

## backup
The purpose of this code is to reliably backup the raw data in a standardized way on the `/baker` archival storage server. In this way we have a frozen snapshot of the data acquisition. Thus, the data can be analyzed/reanalyzed arbitrarily.


`python backup.py` opens the gui to choose folders\
`python backup.py <src> <dest>` Performs the backup from source to destination. 

Will NOT delete the local data.\
The *ONLY* processing done on these files is lossless compression of ephys and video data. 

## ephys_data_to_alf
This code reformats data as it was acquired by skipeglx to be conformant with IBL/ONE like structures.

This will rename the raw data. To prevent mistakes it first checks that the data have been backed up by checking for the presence of a text file that is created once backup has been performed.