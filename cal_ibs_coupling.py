import numpy as np
import scipy.constants
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['axes.formatter.use_mathtext'] = True
import csv
import yaml
import math
import sys

#============ global constants  and beam parameters ================================

pi = scipy.constants.pi
clight = scipy.constants.c

"""
e = scipy.constants.e
m_e = scipy.constants.electron_mass
eps_0 = scipy.constants.epsilon_0
r_e = scipy.constants.value("classical electron radius")
I_A = m_e*clight**3*4.*pi*eps_0/e  # Alfven current.
"""
#--initialize parameters to be used  

c = clight * 100

emit1 = 1.92e-6
emit2 = 1.76e-6

gamma = 107.354
beta  = np.sqrt( 1 - 1/gamma/gamma )

sigmap = 3.5e-4 
sigmas = 93

Nions  = 1.7e9
ru     = 48.971e-16

C = 3.834e5
clight = clight * 100
print( 197, 79, gamma, Nions, C, sigmap, sigmas, emit1, emit2, ru )

#---read in parameters and reset above parameters

f1=open("./beam_parameters.txt", 'r')
lines=f1.readlines()
f1.close()

A     =  float( ( lines[0].split() )[1] )
Z     =  float( ( lines[1].split() )[1] )
gamma =  float( ( lines[2].split() )[1] )
beta  =  np.sqrt( 1 - 1/gamma/gamma )
Nions =  float( ( lines[3].split() )[1] )    
C     =  float( ( lines[4].split() )[1] )
sigmap=  float( ( lines[5].split() )[1] ) 
sigmas=  float( ( lines[6].split() )[1] )
emit1 =  float( ( lines[7].split() )[1] ) 
emit2 =  float( ( lines[8].split() )[1] )
Ninteg=  int( ( lines[9].split() )[1] )   
nturns=  int( ( lines[10].split() )[1] )
filename=( lines[11].split() )[1]

print( A,Z, gamma, Nions, C, sigmap, sigmas, emit1, emit2, Ninteg, nturns, filename )

echarge  = 1.6021765e-19
pmass    = 1.6726216e-27
epsilon0 = 8.854187817e-12
clight   = 2.99792458e8
ru       = ( Z * echarge * Z * echarge ) / ( 4 * pi  *  epsilon0 * A * pmass * clight * clight )

ru     =  ru  * 100
emit1  = emit1  * 100
emit2  = emit2  * 100
C      = C * 100
clight = clight * 100
sigmas = sigmas * 100

print( A,Z, gamma, Nions, C, sigmap, sigmas, emit1, emit2, ru )

#ru     = 48.971e-16

#======================supporting functions=================================

def Rd(x_safe, y_safe, z_safe):
    """Compute RD function, as described in Algorithm 4 of B. C. Carlson, 
    "Computing Elliptic Integrals by Duplication," Numerische Mathematik 33,
    1-16 (1979).

    x_safe, y_safe, and z_safe are the three arguments of the function
    RD(x_safe, y_safe ,z_safe).
    """

    err_tol = 1.e-5

    x = x_safe
    y = y_safe
    z = z_safe

    c1 = 3./7.
    c2 = 1./3.
    c3 = 3./22.
    c4 = 3./11.
    c5 = 3./13.

    s_vec = np.zeros(4)


    first_run = 1
    mu = (x+y+3.*z)/5.
    lamb = np.sqrt(x*y) + np.sqrt(x*z) + np.sqrt(y*z)

    sigma = 0.

    count = 0

    while(max(max(1-x/mu, 1-y/mu), 1-z/mu) > err_tol or first_run):
        first_run = 0


        s_vec[0] = (x**2. + y**2. + 3.*z**2.)/4.
        s_vec[1] = (x**3. + y**3. + 3.*z**3.)/6.
        s_vec[2] = (x**4. + y**4. + 3.*z**4.)/8.
        s_vec[3] = (x**5. + y**5. + 3.*z**5.)/10.

        sigma = sigma + 3. * 4**(-count)/(np.sqrt(z)*(z+lamb))


        x = (x + lamb)/4.
        y = (y + lamb)/4.
        z = (z + lamb)/4.

        mu = (x+y+3.*z)/5.
        lamb = np.sqrt(x*y) + np.sqrt(x*z) + np.sqrt(y*z)
        count = count + 1

    s_vec[0] = (x**2. + y**2. + 3.*z**2.)/4.
    s_vec[1] = (x**3. + y**3. + 3.*z**3.)/6.
    s_vec[2] = (x**4. + y**4. + 3.*z**4.)/8.
    s_vec[3] = (x**5. + y**5. + 3.*z**5.)/10.


    return sigma + 4**(-count)*mu**(-1.5)*(1 + c1*s_vec[0] + c2*s_vec[1]\
            + c3*s_vec[0]**2. + c4*s_vec[2] + c5*s_vec[0]*s_vec[1]\
            + c5*s_vec[3])
 

#========calcluate eign emittance growth contribution from one element==============

def growth_rate( oneline ):
    """callcuate eign emittance growth with one element 
       oneline string from optic input  file
    """

    #print(oneline)
    
    data   = oneline.split()

    index  = data[0]
    name   = data[1]
    s      = data[2]
    beta1x = float( data[3] )  
    alfa1x = float( data[4] ) 
    beta1y = float( data[5] ) 
    alfa1y = float( data[6] ) 
    nu1    = float( data[7] )
    beta2x = float( data[8] ) 
    alfa2x = float( data[9] ) 
    beta2y = float( data[10] )  
    alfa2y = float( data[11])  
    nu2    = float( data[12])
    u      = float( data[13])  
    dx     = float( data[14])
    dxp    = float( data[15]) 
    dy     = float( data[16])
    dyp    = float( data[17])
    q1     = float( data[18])
    q2     = float( data[19])
    m56    = float( data[20])

    #---generate v1 and v2  and  matrix V, U, etc

    if beta1y == 0. or beta2x == 0. or  nu1 == 0. or nu2  == 0. or u == 1. :
        v1=np.array([ np.sqrt(beta1x),  -( 1j * (1-u) + alfa1x )/np.sqrt(beta1x), 0, 0 ])
        v2=np.array([ 0 , 0, np.sqrt( beta2y ),  -( 1j * (1-u) + alfa2y )/np.sqrt(beta2y) ])
    else:
        v1=np.array([ np.sqrt(beta1x),  -( 1j * (1-u) + alfa1x )/np.sqrt(beta1x), np.sqrt( beta1y)* np.exp( 2* pi * 1j * nu1), -(1j * u + alfa1y )* np.exp(2*pi* 1j * nu1)/np.sqrt(beta1y)  ])
        v2=np.array([ np.sqrt(beta2x)*np.exp(2j*pi*nu2), -(1j * u + alfa2x )* np.exp(2j * pi*nu2)/np.sqrt(beta2x), np.sqrt( beta2y ),  -( 1j * (1-u) + alfa2y )/np.sqrt(beta2y) ])

    #print("v1=", v1 )
    #print("v2=", v2 )

    col1=   np.real( v1 )
    col2=  -np.imag( v1 )
    col3=   np.real( v2 )
    col4=  -np.imag( v2 )

    #print(col1, col2, col3, col4)

    V = np.column_stack([col1.T, col2.T, col3.T, col4.T])   #  Eq.(27)

    #print("V=",V)

    #print( np.linalg.det(V) )

    """
    #  directly get V 
    c1= np.cos( 2* pi * nu1 )
    c2= np.cos( 2* pi * nu2 )
    s1= np.sin( 2* pi * nu1 )
    s2= np.sin( 2* pi * nu2 )

    V= np.array([
        [ np.sqrt(beta1x), 0 , np.sqrt(beta2x) * c2,   -np.sqrt(beta2x) *s2 ],
        [ -alfa1x/np.sqrt(beta1x),  (1-u)/np.sqrt(beta1x), (u *s2 - alfa2x  * c2 )/ np.sqrt(beta2x),  ( u * c2 +  alfa2x * s2) / np.sqrt(beta2x) ],
        [np.sqrt(beta1y) * c1,  -np.sqrt(beta1y) * s1 ,   np.sqrt(beta2y), 0 ],
        [ (u*s1-alfa1y*c1)/np.sqrt(beta1y), (u *c1+ alfa1y*s1)/ np.sqrt(beta1y), -alfa2y/ np.sqrt(beta2y), (1-u)/np.sqrt(beta2y) ] 
        ])

    print("V=", V)
    """

    U = np.array( [
        [0 ,1,0,0],
        [-1,0,0,0],
        [0, 0,0,1],
        [0,0,-1,0]
       ] )

    #print("U=",U)

    #print( "V.T @ U @V=U", V.T @ U @ V  )   # Eq.(28)
    #print( "V @ U @ V.T=U",  V @ U @ V.T )   # Eq.(28)

    #-----Beam size  matrix

    Xip =np.array([
             [1/emit1,0,0,0],
             [0,1/emit1,0,0],
             [0,0,1/emit2,0],
             [0,0,0,1/emit2]
        ])

    #print( "Xip= ", Xip )

    Xi = U@V@Xip@(V.T)@(U.T)   # Eq.(30)

    #print( "Xi = ", Xi )

    D=np.array([dx, dxp, dy, dyp ])
    Xi_tot_5 = Xi @ (D.T)
    Xi_tot_55 = 1.0/sigmap/sigmap  + D@Xi@(D.T)

    Xi_tot = np.array([                                         # Eq. (31)
        [Xi[0,0], Xi[0,1], Xi[0,2], Xi[0,3], Xi_tot_5[0] ],
        [Xi[1,0], Xi[1,1], Xi[1,2], Xi[1,3], Xi_tot_5[1] ],
        [Xi[2,0], Xi[2,1], Xi[2,2], Xi[2,3], Xi_tot_5[2] ],
        [Xi[3,0], Xi[3,1], Xi[3,2], Xi[3,3], Xi_tot_5[3] ],
        [Xi_tot_5[0], Xi_tot_5[1], Xi_tot_5[2], Xi_tot_5[3], Xi_tot_55 ]   
        ])

    #print( "Xi_tot =  ", Xi_tot )

    #-----calulate  Coulomb Logrithm  Lc

    sigmax    = np.sqrt( emit1 * beta1x + emit2 * beta2x )
    sigmay    = np.sqrt( emit1 * beta1y + emit2 * beta2y )
    sigmaxtot = np.sqrt( sigmax * sigmax + dx * sigmap * dx * sigmap )
    sigmaytot = np.sqrt( sigmay * sigmay + dy * sigmap * dy * sigmap )
    alfaxy    = ( emit1 *np.sqrt( beta1x * beta1y ) * np.cos( 2*pi*nu1) + emit2 *np.sqrt( beta2x * beta2y ) * np.cos( 2*pi*nu2) ) / ( sigmaxtot * sigmaytot )

    sigma1    = np.sqrt(
                      2*(1-alfaxy*alfaxy) * sigmaxtot * sigmaxtot * sigmaytot * sigmaytot  /
                 ( sigmaxtot * sigmaxtot + sigmaytot * sigmaytot + np.sqrt( (sigmaxtot * sigmaxtot - sigmaytot * sigmaytot)**2 + 4*alfaxy*alfaxy*sigmaxtot * sigmaxtot * sigmaytot * sigmaytot )  )
                       )    
    sigma2    = np.sqrt(
                      2*(1-alfaxy*alfaxy) * sigmaxtot * sigmaxtot * sigmaytot * sigmaytot  /
                 ( sigmaxtot * sigmaxtot + sigmaytot * sigmaytot - np.sqrt( (sigmaxtot * sigmaxtot - sigmaytot * sigmaytot)**2 + 4*alfaxy*alfaxy*sigmaxtot * sigmaxtot * sigmaytot * sigmaytot )  )
                       ) 

    #print( "sigma1, sigma2 = ", sigma1, sigma2 )

    Xi_theta =  np.array([
             [ Xi_tot [1,1], Xi_tot [1,3], Xi_tot [1,4]  ],
             [ Xi_tot [3,1], Xi_tot [3,3], Xi_tot [3,4]  ],
             [ Xi_tot [4,1], Xi_tot [4,3], Xi_tot [4,4]  ]
              ])

    Gamma = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0,0,1.0/gamma]
                    ])

    SigmaThetaBF     = Gamma @ ( np.linalg.inv(Xi_theta) ) @ Gamma
    eigvals, eigvecs = np.linalg.eig( SigmaThetaBF )
    sigmaTheta2BF    = np.array([eigvals[0], eigvals[1], eigvals[2] ])
    sigmav           = gamma * beta * c * np.sqrt(sigmaTheta2BF)
    sigmavtot        = np.linalg.norm ( sigmav  ) 

    #print( "sigmav = ",sigmav )
    #print( "sigmavtot =  ", sigmavtot )

    rmin  = ru * c * c / sigmavtot / sigmavtot    #  no 2 before  ru
    rmaxp = np.sqrt( sigma1 * sigma2 * C * gamma * sigmavtot  *  sigmavtot / 2 / Nions / ru / c / c  )
    rmax  = np.minimum( rmaxp, np.minimum(sigma1, sigma2)  )
    Lc    = np.log( rmax / rmin )

    #print("Lc=  ", Lc )

    #-------\tilde{V1}, \tilde{V2},  Eq.(38)

    v1conjugate  =  np.array( [ v1[0].conjugate(), v1[1].conjugate(), v1[2].conjugate(), v1[3].conjugate() ] )
    v2conjugate  =  np.array( [ v2[0].conjugate(), v2[1].conjugate(), v2[2].conjugate(), v2[3].conjugate() ] )

    V1outer =  np.outer(v1.T, v1conjugate)
    V2outer =  np.outer(v2.T, v2conjugate)

    #print( "V1outer = ", V1outer )
    #print( "V2outer = ", V2outer )

    V1tilde = ( (U.T) @  V1outer @ U ).real
    V2tilde = ( (U.T) @  V2outer @ U ).real

    #print( "V1tilde= ", V1tilde )
    #print( "V2tilde= ", V2tilde )

    #----now we build matrix  A,  Eq. (47) and find out T,  Eq. (50)

    A  =  np.array([
          [ Xi_tot[1,1], Xi_tot[1,3], gamma * Xi_tot[1,4] ],
          [ Xi_tot[3,1], Xi_tot[3,3], gamma * Xi_tot[3,4] ],
          [ gamma*Xi_tot[4,1], gamma * Xi_tot[4,3], gamma * gamma * Xi_tot[4,4] ]    
        ])

    #print("A=  ", A)

    eigvals, eigvecs = np.linalg.eig(A)

    T= np.column_stack([eigvecs[0],eigvecs[1],eigvecs[2]])      

    T = T.T
    
    #print( "eigvals= ", eigvals )
    #print(" eigvecs[0] = ", eigvecs[0] )
    #print(" eigvecs[1] = ", eigvecs[1] )
    #print(" eigvecs[2] = ", eigvecs[2] )

    #print( "T=  ", T )
    #print(" T.T @  T = ", T.T @ T )
    #print("T^t A T  = ", T.T @ A @ T )

    #----now we build matrix  a1, a2, a3,   Eq. (49)

    V1tildeD = V1tilde @ D.T
    a1_33 = gamma*gamma * D @ V1tilde @ (D.T)
    a1 = 0.5 * np.array([
         [ V1tilde[1,1], V1tilde[1,3], gamma * V1tildeD[1] ], 
         [ V1tilde[1,3], V1tilde[3,3], gamma * V1tildeD[3] ],
         [ gamma*V1tildeD[1], gamma*V1tildeD[3], a1_33    ]
        ])

    #print("V1tildeD=  ", V1tildeD)
    #print("a1_33=  ", a1_33 )
    #print("a1=",a1)

    V2tildeD = V2tilde @ D.T
    a2_33 = gamma*gamma * D @ V2tilde @ (D.T)
    a2 = 0.5 * np.array([
         [ V2tilde[1,1], V2tilde[1,3], gamma * V2tildeD[1] ], 
         [ V2tilde[1,3], V2tilde[3,3], gamma * V2tildeD[3] ],
         [ gamma*V2tildeD[1], gamma*V2tildeD[3], a2_33    ]
        ])

    a3 = np.array([
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]      #   in the original paper:  1<---gamma *gamma, should be 1 likely!!!
        ])


    #print( "a1 = ", a1 )
    #print( "a2 = ", a2 )
    #print( "a3 = ", a3 )

    #----now we calculate IBS growth rates

    sigma_a = eigvals

    R1= (1.0/sigma_a[0]) * Rd( 1.0/sigma_a[1],  1.0/sigma_a[2],  1.0/sigma_a[0] )
    R2= (1.0/sigma_a[1]) * Rd( 1.0/sigma_a[2],  1.0/sigma_a[0],  1.0/sigma_a[1] )
    R3= (1.0/sigma_a[2]) * Rd( 1.0/sigma_a[0],  1.0/sigma_a[1],  1.0/sigma_a[2] )

    TtaT = (T.T)@a1@T
    r1 = ( R1 * (np.trace(a1)- 3*TtaT[0,0])  + R2 * (np.trace(a1)- 3*TtaT[1,1]) + R3 * (np.trace(a1)- 3*TtaT[2,2]) ) / np.sqrt(sigma_a[0]* sigma_a[1]*sigma_a[2])
    TtaT = (T.T)@a2@T
    r2 = ( R1 * (np.trace(a2)- 3*TtaT[0,0])  + R2 * (np.trace(a2)- 3*TtaT[1,1]) + R3 * (np.trace(a2)- 3*TtaT[2,2]) ) / np.sqrt(sigma_a[0]* sigma_a[1]*sigma_a[2])
    TtaT = (T.T)@a3@T
    r3 = ( R1 * (np.trace(a3)- 3*TtaT[0,0])  + R2 * (np.trace(a3)- 3*TtaT[1,1]) + R3 * (np.trace(a3)- 3*TtaT[2,2]) ) / np.sqrt(sigma_a[0]* sigma_a[1]*sigma_a[2])

    #print("r1= ", r1)
    #print("r2= ", r2)
    #print("r3= ", r3)

    #----double check Lc based on the paper: another  method
    """
    Sigmav  = (gamma * beta * c ) **2  * np.linalg.inv( A )
    rhomin  = ru * c * c / np.trace( Sigmav )

    Simga    = np.linalg.inv( Xi_tot  )
    sigmamin = np.sqrt( ( Simga[0,0] + Simga[2,2]  - np.sqrt( (Simga[0,0] - Simga[2,2] )**2 + 4 * Simga[0,2] *  Simga[0,2] ) ) / 2 )

    rhomax = np.minimum( sigmamin, np.minimum(gamma* sigmas, np.sqrt( np.trace(Sigmav)/4/pi/(Nions/C)/ru/c/c) )  )

    Lc= np.log( rhomax/rhomin )

    #print("Sigmav =", Sigmav )

    #print("Lc=  ", Lc )
    """

    #------final

    return r1,r2,r3, Lc


#========calcluate eign emittance growth contribution from one element==============

def growth_total_rate():
    """callcuate eign emittance growth with one element 
       oneline string from optic input  file
    """

    f1=open(filename, 'r')
    holder= f1.readlines()
    f1.close()

    sum1 =0
    sum2 =0
    sum3 =0

    #for i in range( len(holder)-2 ):
    for i in range( len(holder)-2 ):
        oneline = holder[i+1]
        data    = oneline.split()
        index   = data[0]
        name    = data[1]
        s1      = float( data[2]  )

        s2      = (holder[i+2]).split()
        s2      = float( s2[2] )
        dl      = s2-s1

        r1,r2,r3,Lc = growth_rate( oneline )

        sum1 = sum1  + Lc * r1 * dl
        sum2 = sum2  + Lc * r2 * dl
        sum3 = sum3  + Lc * r3 * dl   

        #print(index, name, s1, "r1, r2, r3 Lc: ", r1, r2, r3, Lc )

    scale_t    =  Nions * ru * ru * c /( 6 * pi * sigmas * beta**3 * gamma**4 * emit1 * emit2 * sigmap ) / C
    scale_l    =  Nions * ru * ru * c /( 12 * pi * sigmas * beta**3 * gamma**2 * emit1 * emit2 * sigmap ) / C
    demit1dt   =  sum1 * scale_t
    demit2dt   =  sum2 * scale_t
    demit3dt   =  sum3 * scale_l 

    return demit1dt, demit2dt, demit3dt

    
#=============================================
#    
#   main()
#
#=============================================

dt    = ( C / clight / beta  ) * nturns

"""
print( C)
print( clight )
print( nturns, dt  )
print( Ninteg, Ninteg*dt)
sys.exit()
"""

 
f2=open("./output.dat", 'w')
f2.write("time(h) emit1(m)  emit2(m)  sigmas(m)  sigmap  tao1(min)  tao2(min)  tao3(min) \n")

demit1dt, demit2dt, demit3dt = growth_total_rate()
tao1  = 1.0/( demit1dt/emit1 ) / 60
tao2  = 1.0/(demit2dt/emit2 ) / 60
tao3  = 1.0/(demit3dt/sigmap/sigmap ) / 60
print( 0, emit1, emit2, sigmas, sigmap, 1.0/( demit1dt/emit1 ) / 3600, 1.0/(demit2dt/emit2)/ 3600,   1.0/(demit3dt/sigmap/sigmap ) / 3600  )
f2.write(f"{0.:12.6e}  {emit1/100:12.6e}  {emit2/100:12.6e}  {sigmas/100:12.6e}  {sigmap:12.6e}  {tao1:12.6e}  {tao2:12.6e}  {tao3:12.6e}    \n")  

for  i in range( Ninteg ):
    demit1dt, demit2dt, demit3dt = growth_total_rate()
    tao1  = 1.0/( demit1dt/emit1 ) / 60
    tao2  = 1.0/(demit2dt/emit2 ) / 60
    tao3  = 1.0/(demit3dt/sigmap/sigmap ) / 60 
    emit1 = emit1 + demit1dt  *  dt
    emit2 = emit2 + demit2dt  *  dt
    sigmap= sigmap  +  ( demit3dt /2 / sigmap )  * dt
    sigmas= sigmas  +  sigmas * ( demit3dt / 2 / sigmap / sigmap ) *  dt
    """
    if dt *(i+1)/3600. < 0.4:
        Nions = (15.949635e9 / 28 ) * np.exp( - dt *(i+1)/3600. / 16.841416 )
    else:
        Nions = ( 16.358151e9 / 28 ) * np.exp( - dt *(i+1)/3600. / 8.639165 )
     """
    
    print( dt *(i+1)/3600., emit1/100, emit2/100, sigmas/100, sigmap, tao1, tao2, tao3 )
    f2.write(f"{dt *(i+1)/3600.:12.6e}  {emit1/100:12.6e}  {emit2/100:12.6e}  {sigmas/100:12.6e}  {sigmap:12.6e}  {tao1:12.6e}  {tao2:12.6e}  {tao3:12.6e}    \n") 
f2.close()

