c 
c     lowmufit_sk4.F         17-SEP-2008     Y.Takeuchi
c
c     for SK-IV process apply both lowe & muon fitters
c
c   output: (arg1)   lomu file (skroot)
c   input:  (arg2-)  ofl file/subrun (ZBS)
c
c /disk/data7/sk4/zbs/0607/060739/ofl_run060739.000001
c-rw-r--r--  1 takeuchi sk 44895083  lomu_lomutq.root  lo TQ & mu TQ
c-rw-r--r--  1 takeuchi sk 37806460  lomu_lotq.root    only lo TQ
c-rw-r--r--  1 takeuchi sk  7468494  lomu_mutq.root    only mu TQ
c-rw-r--r--  1 takeuchi sk   378836  lomu_notq.root    no TQ
c-rw-r--r--  1 takeuchi sk  6172604  lo TQ & mu TQ only in 1.3 micro sec.
c
      implicit none

#include "skhead.h"
#include "skparm.h"
#include "sktq.h"
#include "skbadc.h"
#include "geopmt.h"
#include "skday.h"
#include "skwt.h"
#include "skroot_lowe.h"
#include "lininfo2.h"      
#include "lfparm.h"
      integer iret, lfflag, ipbank, i, lenchr, nread, nevent(20)
      real    watert, rver
      integer icall, isave_event,iday, lastrun, log_level,iskip

      integer lun
      parameter (lun=10)

      integer iargc, narg
      character*200 fname_in
      character*200 fname_out
      character*10 nsub,tmp,nrun
      character tentative
      character*200 fTentativeConfig
      logical ntentative
      real watertTentative, betaTentative
c*** number of hit limit for clusfit
      integer NHIT_CUT
      parameter (NHIT_CUT = 400)

      integer lecompte,n,nnrun
      real px, py, pz,width

cccccccccccccccccccccccccccccccccccccccccccccccccc

      character*200 temp
      integer it,nintoskip,nouttoskip
      parameter (nintoskip=6)
      parameter (nouttoskip=6)
      character*200 in_branches_to_skip(nintoskip)
      character*200 out_branches_to_skip(nouttoskip)

      data in_branches_to_skip /'SPACERS','QBEESTATUS','DBSTATUS','MISMATCHEDHITS',
     $     'ATMPD','UPMU'/
      data out_branches_to_skip /'TQLIST','ODTQLIST','SOFTWARETRG','EVENTTRAILER',
     $     'HWTRGLIST','PEDESTALS'/

      flag_g4_wtpar = 1 ! for SKG4
      ! flag_g4_wtpar = 0 ! for detsim

c*** check arguments
      narg = iargc()
      if (narg .lt. 2) then
         print *, 'Usage: lowfit_data [runnum] [output file] [input file] [?tentative?] [?config]'
         call exit(1)
      endif

c*** get LINAC run number
      call getarg(1, nrun)
      read(nrun,*)nnrun

c*** get output file name
      call getarg(2, fname_out)
      call skroot_open(lun,fname_out(1:lnblnk(fname_out)))

c***  prepare input files
      do i = 3,3
         call getarg(i, fname_in) 
         call skroot_set_input_file(lun,fname_in(1:lnblnk(fname_in)))
      enddo
#ifdef TENTATIVE
c***  tentative paramter
      call getarg(4, tentative)
      print *, 'tentative is ', tentative
      read(tentative,*)ntentative
      if (ntentative) then
         call getarg(narg, fTentativeConfig)
         open(unit=1996, file=fTentativeConfig)
         read(1996,*)watertTentative
         print *, 'tentative watert = ', watertTentative
      endif
#endif
c*** skip any unneeded branches; this must be called before skroot_init
      do it=1,nintoskip
         temp = in_branches_to_skip(it)
         call skroot_zero_branch(lun,0,temp(1:lnblnk(temp)))
      enddo

      do it=1,nouttoskip
         temp = out_branches_to_skip(it)
         call skroot_zero_branch(lun,1,temp(1:lnblnk(temp)))
      enddo

c*** initilalize skroot files
      call skroot_init(lun)

c*** set SK_GEOMETRY
      SK_GEOMETRY = 6
      ! SK_GEOMETRY = 5

c*** init.
      call kzinit
      call geoset
      call skbadopt(55)
      call skoptn('31,30,25')

c***  init water transparency table
      call skrunday 
      call skwt_gain_corr 
c      call skwt 
      print *,'relapse(nnrun) = ',relapse(nnrun)

c***  check lininfo
      call linsum2(nnrun)
      print *, 'lininfo2:', lin_mode(nnrun),lin_trg(nnrun),
     $     lin_x(nnrun),lin_y(nnrun),lin_z(nnrun),
     $     lin_badrun(nnrun), lin_runnum, lin_mom(nnrun),
     $     lin_dark, darkfac
         
      call skbadch(lin_badrun(nnrun), 1, iret)
      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c*** main loop 
 10   CONTINUE 
      isave_event = 0
      CALL SKRAWREAD(lun,*1002,*1001,*10,*10)
      CALL SKREAD(-lun,*1002,*1001,*10,*10)

      nread = nread + 1
      if(mod(nread,1000).eq.0) print *,nread,nrunsk,nevsk,nqisk,nevent

c***  obtain watert and badch
      if (nnrun .ne. lastrun) then
         call lfwater(relapse(lin_badrun(nnrun)), watert)
         print *, 'nnrun, lin_badrun, watert = ',
     $        nnrun, lin_badrun(nnrun), watert
         lastrun = nnrun
      endif
#ifdef TENTATIVE
      if (ntentative) then
         watert=watertTentative
      endif
#endif
c*** init. bonsai 
      if (icall.eq.0) then
         CALL cfbsinit(MAXPM,xyzpm)
         icall = 1
      endif

c*** load all branches
      call skroot_get_entry(lun)

c*** remove pedestal
      if (btest(ifevsk,9)) goto 10
      i = 1
      nevent(i) = nevent(i) + 1
c*** remove spacer 
      if (btest(ifevsk,19)) goto 10
      i = i + 1
      nevent(i) = nevent(i) + 1
c*** remove LED burst
      if (btest(ifevsk,26)) goto 10
      i = i + 1
      nevent(i) = nevent(i) + 1
c***  if Relic trigger is on, stop it
      i = i + 1
      if (btest(idtgsk,29) ) then
         nevent(i) = nevent(i) + 1
         print*,'********************************'
         print*,'  After trigger? it is strange'
         print*,'********************************'
         stop
      endif
c*** require SLE or LE or HE or LINAC or LINAC-RF
c     if (.not.btest(idtgsk,0) .and. .not.btest(idtgsk,1) 
c    $     .and. .not.btest(idtgsk,24) 
c    $     .and. .not.btest(idtgsk,25)) goto 10
      if(.not.btest(idtgsk,24)) goto 10
c     if(.not.btest(idtgsk,1)) goto 10
      i = i + 1
      nevent(i) = nevent(i) + 1
            
c*** skip incomplete
      if (iand(ifevsk, 2**20).ne.0) goto 10
      i = i + 1
      nevent(i) = nevent(i) + 1


c***  shift timing gate +1000nsec for LINAC only or LINAC-RF only
      if (.not.btest(idtgsk,0) 
     $     .and. (btest(idtgsk,24) .or. btest(idtgsk,25))) then
         call set_timing_gate_nsec(1000.0)
         call skread(-lun,*1003,*1003,*1003,*1003)
      endif

c*** clear variables
      call lfclear_all()

c*** store lnqisk (will be rewrite in lfallfit_sk4_data)
c*** linfo=1: nflf
      call lfflasher(lmx24, qimxsk, nflf)
c*** linfo=2: lnahit
      call lfnhita(lnahit)
c*** linfo=3: lnqisk
      lnqisk = nqisk
      lninfo = 3 
         
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c*** for lowe events
      if (nqisk.lt.2000.) then
         i = i + 1
         nevent(i) = nevent(i) + 1
c*** apply lowfit
         i = i + 1
         if (nqisk.lt.NHITCUT) then
            nevent(i) = nevent(i) + 1
            call lfallfit_sk6_data(watert, NHITCUT, iskip, log_level, lfflag)
            ! call lfallfit_sk5_data(watert, NHITCUT, iskip, log_level, lfflag)
         endif
c*** save event flag
         isave_event = 1
      else
         i = i + 2
      endif

c***  for select single electron event         
      n=lecompte(lin_x(nnrun),lin_y(nnrun),lin_z(nnrun),width)
      if(n .ne. 1)then
c         print *,'n,nevsk,nqisk',n,nevsk,nqisk,nread
         goto 10
      end if

      rver = sqrt((lin_x(nnrun) - bsvertex(1))**2 + (lin_y(nnrun) - bsvertex(2))**2+ (lin_z(nnrun) - bsvertex(3))**2)
      if(rver.gt.300)then
         print *,'rver is big',rver,nevsk,nqisk,nread
         goto 10
      end if

c      if(bsgood(2).lt.0.4)then
c         goto 10
c      end if
      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c*** output skroot file
      i = i + 1
      if (isave_event .ne. 0) then
         nevent(i) = nevent(i) + 1
c         if(nread.ge.100000) go to 9999
c***  store LOWE branch
         call skroot_set_lowe(lun,bsvertex, bsresult, bsdir, 
     $        bsgood, bsdirks, bseffhit, bsenergy, bsn50, bscossun,
     $        clvertex, clresult, cldir, clgoodness,     
     $        cldirks, cleffhit, clenergy, cln50, clcossun,
     $        latmnum, latmh, lmx24, ltimediff, lnsratio, lsdir, 
     $        spaevnum, spaloglike, sparesq, spadt, spadll, spadlt,
     $        spamuyn, spamugdn,
     $        posmc, dirmc, pabsmc, energymc, darkmc, islekeep,
     $        bspatlik, clpatlik, lwatert, lninfo, linfo)
               
c*** remove hits outside 1.3microsec
         call delete_outside_hits

c***  store header & TQ info.
         call skroot_set_tree(lun)

c***  output root file 
         call skroot_fill_tree(lun) ! write to disk

      endif

      goto 10

 1003 continue
      write(0,*)'read error2'
      stop
 1002 continue
      write(0,*)'read error'
      goto 10
 1001 continue

 9999 continue
c*** close output file
      call skroot_close(lun)
      call skroot_end

c*** for bonsai 
      call cfbsexit
      print *, 'done. nrun/nsub/nread/nevent=', 
     $     nrunsk, nsubsk, nread, nevent

      stop
      end
#include "lecompte.F"
#include "linsum2.F"
