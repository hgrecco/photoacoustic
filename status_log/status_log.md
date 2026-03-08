
**Date:** 2026-03-08 17:04:13

All figures are built correctly now. 
The issue before was that watchdog's observer's queue wass filling up and dropping some of the events on it.
for that reason, some of the files created were ignored and never processed.

Now I have other issues: should move the experiment update functionality to the event handler and be sure that no analysis or plotting function is called if the created file is not a data file (including absorbance).
Right now, the copied .fl files are creating issues on the analysis bc some processing funcitons are being called at the wrong time.

**Date:** 2026-03-07 02:44:23

currently all figures but the linear are built correctly. 
linear fit looks like this:

![alt text](image.png)

there are like 4 linear plots. Colors are weird and it doesn't match the original analysis.