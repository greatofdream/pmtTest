      FUNCTION lecompte(px,py,pz,width)
         IMPLICIT NONE
#include <skparm.h>
#include <sktq.h>
#include <geopmt.h>
         REAL      darkthreshold,minthreshold
         REAL      multistrength,multithreshold
         PARAMETER (darkthreshold=4,minthreshold=8)
         PARAMETER (multistrength=0.75,multithreshold=5)

         INTEGER lecompte
         REAL px,py,pz,width
         INTEGER hit1,hit2,cab,gcount
         REAL    dx,dy,dz,dr,ttof(MAXPM),dt2,g,good(MAXPM)
         REAL    tsum,tsum2,gmax,gthreshold

         DO hit1=1,nqisk
            cab=ihcab(hit1)
            dx=xyzpm(1,cab)-px
            dy=xyzpm(2,cab)-py
            dz=xyzpm(3,cab)-pz
            dr=sqrt(dx*dx+dy*dy+dz*dz)
            ttof(hit1)=tisk(cab)-dr/21.58333
            good(hit1)=0
         END DO         
         DO hit1=1,nqisk
            DO hit2=hit1+1,nqisk
               dt2=0.2*(ttof(hit1)-ttof(hit2))
               dt2=dt2*dt2
               IF (dt2<25) THEN
                  g=exp(-0.5*dt2)
                  good(hit1)=good(hit1)+g
                  good(hit2)=good(hit2)+g
               END IF
            END DO
         END DO
         gcount=0
         gmax=0
         DO hit1=1,nqisk
            IF (good(hit1) .GE. darkthreshold) THEN
               gcount=gcount+1
            END IF
            IF (good(hit1) .GT. gmax) THEN
               gmax=good(hit1)
            END IF
         END DO
         IF (gcount .LE. 2) THEN
            lecompte=0
            RETURN
         END IF
         gthreshold=multistrength*gmax-
     -              3*sqrt(multistrength*gmax)
         IF (gthreshold .LT. minthreshold) THEN
            gthreshold=minthreshold
         END IF
         gcount=0
         tsum=0
         tsum2=0
         DO hit1=1,nqisk
            IF (good(hit1) .GE. gthreshold) THEN
               gcount=gcount+1
               dt2=ttof(hit1)-800
               tsum=tsum+dt2
               tsum2=tsum2+dt2*dt2
            END IF
c            print *,ttof(hit1),good(hit1)
         END DO
c         print *,'nqisk=',nqisk,gcount,gthreshold
c         DO hit1=1,nqisk
c            print *,ttof(hit1),good(hit1)
c         END DO
         IF (gcount .LE. 1) THEN
            lecompte=1
            RETURN
         END IF
         tsum=tsum/gcount
         width=sqrt(tsum2/gcount-tsum*tsum)
         IF (width .LE. multithreshold) THEN
            lecompte=1
         ELSE
            lecompte=2
         END IF
c         print *,'found ',lecompte,' hits'
         RETURN
      END
