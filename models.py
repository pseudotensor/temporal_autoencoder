
#######################################################
#
# Code from Ilya Sutskever at
# http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
#
#######################################################


from pylab import *
import tensorflow as tf 


#FLAGS = tf.app.flags.FLAGS

def norm(y): return sqrt((y**2).sum())
def sigmoid(y): return 1./(1.+exp(-y))


SIZE=10

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

# size of bounding box: SIZE X SIZE.

def model_n(T=64, TY=0, n=2, r=None, m=None):
    if r is None: r=array([4.0]*n)
    if m is None: m=array([1]*n)
    # r is to be rather small.

    X=zeros((T, n, 2), dtype='float')
    V = zeros((T, n, 2), dtype='float')
    if TY==0:
        v = randn(n,2)
        v = (v / norm(v)*.5)*1.0
    else:
        v=0*randn(n,2)

    good_config=False

    goodconfigattempt=0
    maxgoodconfigattempts=10
    while not good_config:
        goodconfigattempt=goodconfigattempt+1
        if goodconfigattempt>maxgoodconfigattempts:
            break
        
        x = 2+rand(n,2)*8
        good_config=True
        for i in range(n):
            for z in range(2):
                if x[i][z]-r[i]<0:      good_config=False
                if x[i][z]+r[i]>SIZE:     good_config=False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i]-x[j])<r[i]+r[j]:
                    good_config=False
                    
    
    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t,i]=x[i]
            V[t,i]=v[i]
            
        for mu in range(int(1/eps)):


            for i in range(n):
                #x[i]+=eps*v[i]
                x[i]+=.5*v[i]
            
            

            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<0:  v[i][z]= abs(v[i][z]) # want positive
                    if x[i][z]+r[i]>SIZE: v[i][z]=-abs(v[i][z]) # want negative
            for i in range(n):
                    for j in range(i):
                        if norm(x[i]-x[j])<r[i]+r[j]:
                            # bouncing off:
                            w    = x[i]-x[j]
                            w    = w / norm(w)
  
                            v_i  = dot(w.transpose(),v[i])
                            v_j  = dot(w.transpose(),v[j])
  
                            new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)
                         
                            v[i]+= w*(new_v_i - v_i)
                            v[j]+= w*(new_v_j - v_j)
  


    return X, V

def ar(x,y,z):
    return z/2+arange(x,y,z,dtype='float')

def tomatrix(X,V,res,TY=0,r=None):

    T, n= shape(X)[0:2]
    if r is None: r=array([4.0]*n)

    mat=zeros((T,res,res, 3), dtype='float')
    
    [I, J]=meshgrid(ar(0,1,1./res)*SIZE, ar(0,1,1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            if TY==0:
                # ball
                ball=exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )
            else:
                # rotating disk
                xx=(I-X[t,i,0])
                yy=(J-X[t,i,1])
                radius=np.sqrt(xx**2+yy**2)
                theta=np.arctan2(xx,yy)
                size=r[i]*3
                ball=radius<size
                omega=0.1
                ball=ball*(np.sin(theta+omega*t))**2
                    
            mat[t, :, :, 1] += 0.0 * (1.0               ) * ball # Green
            mat[t, :, :, 0] += 0.0 * (0.0*V[t,i,0] + 1.0) * ball # Blue
            mat[t, :, :, 2] += 1.0 * (0.0*V[t,i,1] + 1.0) * ball # Red
            
        # truncate if Velocity leads to larger than 1, so can map to 0..255 scale normally
        mat[t,:,:,0][mat[t,:,:,0]>1]=1
        mat[t,:,:,1][mat[t,:,:,1]>1]=1
        mat[t,:,:,2][mat[t,:,:,2]>1]=1
    return mat

def model_vec(res, n=2, T=64, TY=0, r =None, m =None):
    if r is None: r=array([1.2]*n)
    x,v = model_n(T,TY,n,r,m);
    V = tomatrix(x,v,res,TY,r)
    return V

def generate_model_sample(batch_size, seq_length, shape, num_balls, type_balls):
  dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  for i in xrange(batch_size):
    dat[i, :, :, :, :] = model_vec(shape, num_balls, seq_length, type_balls)
  return dat 
