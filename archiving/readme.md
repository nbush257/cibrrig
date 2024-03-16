# Archiving

This is still a WIP and not functioning to any degree. 
Best practice currently is to copy manually.


The purpose of this code is to reliably backup the raw data in a standardized way on the `/baker` archival storage server. In this way we have a frozen snapshot of the data acquisition. Thus, the data can be analyzed/reanalyzed arbitrarily.

The *ONLY* processing done on these files is lossless compression of ephys and video data. 

## Structure

With inspiration from the IBL, recordings are to be saved in the following format:
`{root}/Subjects/{subjectID}/{date: YYYY-MM-dd}/{session}`
where root is some folder on the Archival Baker RSS
e.g.:
`/baker/ramirez_j/ramirezlab/cibrrig/Subjects/m2024-01/2024-02-03/001`


A "session" is defined as a spikeglx "gate"

TODO: make subject json with fields:
