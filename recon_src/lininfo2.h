ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	lininfo.h
c       Original was written by H.Ishino
c       1999/4/21 Add 'lin_mom' by N.Sakurai
c       2003/5/17 Add 'lin_dark' by Y.Koshio
C       2009/10/7 increase lin_num to 80000 for SK-IV  by M. Nakahata
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	integer lin_num
	parameter(lin_num=150000)
	integer lin_mode(lin_num),lin_trg(lin_num)
	real lin_x(lin_num),lin_y(lin_num),lin_z(lin_num)
	integer lin_badrun(lin_num)
	integer lin_runnum
        real lin_mom(lin_num), lin_dark, darkfac

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c	lin_trg    0    linac + normal trigger
c	           1    micro wave trigger
c	           2    clock trigger
c 	           3    only linac trigger 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	common/linacinfo/lin_mode,lin_x,lin_y,lin_z,
     &      lin_badrun, lin_trg, lin_mom, lin_dark
	common/linacinfox/lin_runnum
