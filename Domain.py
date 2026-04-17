import config
import numpy as np
import tensorflow as tf
    
import os
import vtk 
from vtk.util import numpy_support as vns

from vtk.util.numpy_support import vtk_to_numpy







class Interval:
    def __init__(self, ll, rr):
        self._ll = ll
        self._rr = rr

    @property
    def left(self):
        return self._ll

    @property
    def right(self):
        return self._rr
    
    def generate_points(self, N):
        return (self._rr - self._ll) * np.random.rand(N) + self._ll

    def uniform_points(self, N, endpoint=True):
        if endpoint:
            return np.linspace(self._ll, self._rr, num=N, dtype=config.real(np))[:, None]
        else:
            return np.linspace(self._ll, self._rr, num=N + 1, endpoint=False, dtype=config.real(np))[1:, None]
    
    def log_uniform_points(self, N, base, pow_min, endpoint=True):
        v0 = np.linspace(0.0, 0.0, num=1, endpoint=True, dtype=config.real(np))
        if endpoint:
            v1 = np.logspace(pow_min, 0.0, num = N-1, base=base, endpoint=True, dtype=config.real(np))[:, None]
            v1 = v1 * self._rr + self._ll
            return np.append(v0,v1)
        else:
            v1 = np.logspace(pow_min, 0.0, num = N, base=base, endpoint=False, dtype=config.real(np))[1:, None]
            v1 = v1 * self._rr + self._ll
            return np.append(v0,v1)

class Space_1D(Interval):
    def __init__(self, xmin, xmax):
        super().__init__(xmin, xmax)

    @property
    def xmin(self):
        return self._ll
    
    @property
    def xmax(self):
        return self._rr
    
    def cos_uniform_theta(self,N):
        cos_min =np.cos(self._rr)
        cos_max =np.cos(self._ll)
        
        cos_theta = np.random.uniform(cos_min, cos_max, size=N).astype(config.real(np))
        theta = np.arccos(cos_theta)
        return theta
        

class TimeDomain(Interval):
    def __init__(self, tmin, tmax):
        super().__init__(tmin, tmax)

    @property
    def tmin(self):
        return self._ll
    
    @property
    def tmax(self):
        return self._rr
    


class DomainSphere(Interval):
    # def __init__(self, rDomain, thetaDomain, phiDomain, timeDomain, a2=0.4):
    def __init__(self, rDomain, thetaDomain, timeDomain, a2=0.4):
    
        self.rDomain = rDomain
        self.qDomain = thetaDomain
        # self.pDomain = phiDomain
        self.timeDomain = timeDomain
        self.a2 = a2
        
        Rmax = float(1.0 + self.a2*2.0)
        self.xmin = -Rmax
        self.xmax =  Rmax
        self.ymin = -Rmax
        self.ymax =  Rmax
        self.zmin = -Rmax
        self.zmax =  Rmax
        self.tmin = self.timeDomain.tmin
        self.tmax = self.timeDomain.tmax
        theta_min = self.qDomain.left
        theta_max = self.qDomain.right
        # phi_min   = self.pDomain.left
        # phi_max   = self.pDomain.right
    
    ####    デカルト変換     ####
    def sphere_to_cartesian(self, r, theta):
        x = r * np.sin(theta) 
        y = r * np.cos(theta) 
        return x, y
    
    def R_theta(self, theta):
        return 1.0 + self.a2 * (3.0*np.cos(theta)**2 - 1.0)

    def PHI(self, r, theta):
        return r - self.R_theta(theta)


    def genResidualPoints(self, Nf, Nt, out_dir):

        time_list = np.linspace(0.0, self.tmax, Nt)
        all_data = []
        os.makedirs(out_dir, exist_ok=True)

        for t_fixed in time_list:
            
            # 液滴の範囲を変えたいときはここをかえる
            # phi = u1, theta = u2 
            #  u1 = np.random.uniform(0.0, 0.05-1e-6, size=Nf).astype(config.real(np))
             u2 = np.random.uniform(0.0, 1.0, size=Nf).astype(config.real(np))
             u3 = np.random.rand(Nf)
            
            #  phi   = 2.0*np.pi*u1
             theta = np.arccos(2.0*u2 - 1.0)

             
             P2 = (3.0 * np.cos(theta)**2 - 1.0)
             cos_t = np.cos(t_fixed)

            #  Rmax_theta = self.R_theta(theta) * cos_t

             Rmax_theta = 1.0 + self.a2 * P2 * cos_t
            
             r = np.cbrt(u3) * Rmax_theta
             x, y= self.sphere_to_cartesian(r, theta)
             t_array = np.full(Nf, t_fixed)
             data = np.column_stack((x, y, t_array))
             all_data.append(data)

        GE_points = np.vstack(all_data)
        
        # print(f"x.shape: {x.shape}")
        # print(f"y.shape: {y.shape}")
        # print(f"z.shape: {z.shape}")
        # print(f"Total points: {GE_points.shape[0]}")
        # print("\nGE_points shape:", GE_points.shape)        
        # print("\nSaved TSV:GE_points.tsv") 
        # print(type(GE_points))
        # print(GE_points.shape)
        # print(GE_points.ndim)
        np.savetxt('{:}/GE_points.tsv'.format(out_dir), GE_points, fmt='%16.8e', delimiter='\t', header='x\ty\tz\tt', comments='')
        return tf.convert_to_tensor(all_data, config.real(tf))

    
    def split_tsv_by_time(self, input_file, out_dir, tol=1e-8):

        os.makedirs(out_dir, exist_ok=True)

    # 読み込み（ヘッダあり想定）
        data = np.loadtxt(input_file, skiprows=1)

    # 列分解
        x = data[:, 0]
        y = data[:, 1]
        # z = data[:, 2]
        t = data[:, 2]

    # tのユニーク値（丸めて誤差対策）
        t_unique = np.unique(np.round(t, 8))

        print(f"Total unique time steps: {len(t_unique)}")

    # 各tごとに分割
        for i, t_val in enumerate(t_unique):

            mask = np.abs(t - t_val) < tol
            data_t = data[mask]

            filename = f"{out_dir}/split_{i:05d}.tsv"

            np.savetxt(filename,
                       data_t,
                       fmt='%16.8e',
                       delimiter='\t',
                       header='x\ty\tz\tt\tu\tv\tw\tp',
                       comments='')

            print(f"Saved: {filename} (t={t_val:.5f}, N={data_t.shape[0]})")


    def genBoundaryPoints(self,Nbc,Nt,out_dir):
        os.makedirs(out_dir, exist_ok=True)
        time_list = np.linspace(0.0, self.tmax, Nt)
        all_data = []

        for t_fixed in time_list:
            #  u1 = np.random.uniform(0.0, 1.0-1e-6, size=Nbc).astype(config.real(np))
             u2 = np.random.uniform(0.0, 1.0, size=Nbc).astype(config.real(np))

             theta = np.arccos(2.0*u2 - 1.0)
           
             P2 = (3.0 * np.cos(theta)**2 - 1.0)
             cos_t = np.cos(t_fixed)
             
             Rmax_theta = 1.0 + self.a2 * P2 * cos_t

             r = Rmax_theta
             x,y = self.sphere_to_cartesian(r, theta)
             t_array = np.full(Nbc, t_fixed)
             data = np.column_stack((x, y, t_array))
             all_data.append(data)
        
        BC_points = np.vstack(all_data)
        np.savetxt('{:}/BC_points.tsv'.format(out_dir), BC_points, fmt='%16.8e', delimiter='\t', header='x\ty\tz\tt', comments='')
        return tf.convert_to_tensor(BC_points, config.real(tf))
            

    def genGirdPoints(self, Nr, Nq, Nt, out_dir):
        
        os.makedirs(out_dir, exist_ok=True)
        
        t_list     = np.linspace(0.0, self.tmax, Nt)
        theta_list = np.linspace(self.qDomain.left, self.qDomain.right, Nq)
        r_list     = np.linspace(0.0, 1.0, Nr)
        
        data = []
        
        for t_fixed in t_list:
            
            cos_t = np.cos(t_fixed)
            
            # for phi in phi_list:
            for theta in theta_list:
                    P2 = (3.0 * np.cos(theta)**2 - 1.0)
                    Rmax_theta = 1.0 + self.a2 * P2 * cos_t
                    
                    for r_theta in r_list:
                        r = r_theta * Rmax_theta
                        x, y = self.sphere_to_cartesian(r, theta)
                        data.append([x, y, t_fixed])
              
    
        data = np.array(data)
        Grid_points = np.array(data)
        np.savetxt('{:}/Grid_points.tsv'.format(out_dir), Grid_points, fmt='%16.8e', delimiter='\t', header='x\ty\tz\tt', comments='')
        print("出力完了")
        return tf.convert_to_tensor(Grid_points    , config.real(tf))

                    
    
    def tsv_to_vtu_timeseries(self, tsv_file, out_dir):

        os.makedirs("output_test", exist_ok=True)
        os.makedirs("output_vtu", exist_ok=True)
        os.makedirs("output_vtu_BC", exist_ok=True)
        os.makedirs("output_vtu_Grid", exist_ok=True)
        
        #TSV読み込み
        data = np.loadtxt(f"output_test/{tsv_file}", skiprows=1)
        x = data[:,0]
        y = data[:,1]
        # z = data[:,2]  
        t = data[:,2]


        t_rounded = np.round(t, 5)
        unique_t = np.unique(t_rounded)
        # unique_t = np.unique(t)
        print('number timesteps:', len(unique_t))

        for i, ti in enumerate(unique_t):

            mask = np.isclose(t_rounded, ti)
            
            x_i = x[mask]
            y_i = y[mask]
            # z_i = z[mask]

            pts_np = np.column_stack((x_i, y_i))
            N = pts_np.shape[0]

            # == VTU 書き出し ==
            points = vtk.vtkPoints()    #点のからデータ格納
            points.SetData(vns.numpy_to_vtk(pts_np)) #numpy配列をVTKの形式に変換
            grid = vtk.vtkUnstructuredGrid()         #vtuファイルの入れ物作成
            grid.SetPoints(points)                   #pointをセット   

            cells = vtk.vtkCellArray()
            
            print(f"time={ti}, N={np.sum(mask)}")
            
            for j in range(N):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0,j)
                cells.InsertNextCell(vertex)

            grid.SetCells(vtk.VTK_VERTEX, cells)

            t_array = vns.numpy_to_vtk(np.full(N, ti))
            t_array.SetName("time")
            grid.GetPointData().AddArray(t_array)

            filesname = f"{out_dir}/result_{i:04d}.vtu"

            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(filesname)
            writer.SetInputData(grid)
            writer.Write()
            print("Saved:", filesname)
    
    

if __name__ == "__main__":
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'
    rDomain     = Space_1D(1e-6, 1.0)
    thetaDomain = Space_1D(0.0, np.pi)
    # phiDomain   = Space_1D(0.0, np.pi)
    timeDomain = TimeDomain(0.0, 2*np.pi)
    domain = DomainSphere(rDomain, thetaDomain, timeDomain, a2=0.1)    # 点の数
    Nf  = 10000   # 内部点
    Nt  = 100
    out_dir = "output_test"
    GE_points = domain.genResidualPoints(Nf, Nt, out_dir=out_dir)
    Grid_points = domain.genGirdPoints(Nr=5, Nq=50, Nt=30, out_dir=out_dir)
    domain.split_tsv_by_time(input_file='output_test/Grid_points.tsv', out_dir="output_split_Grid")
    
     
     
     
     
     
     
     
     
    # def genResidualPoints_Ad(self, Nf, Nt):
    #     # time_list = self.timeDomain.linspace(0.0, self.timeDomain.right, Nt)
    #     time_list = np.linspace(0.0, self.timeDomain.right, Nt)
    #     # time_list = self.timeDomain.linspace(Nt)
    #     all_data = []
    #     os.makedirs(out_dir, exist_ok=True)
        
    #     for t_fixed in time_list:
            
    #         phi   = self.pDomain.uniform_points(Nf).astype(config.real(np))   # phi
    #         theta = self.qDomain.cos_uniform_theta(Nf).astype(config.real(np))   # theta
    #         u3 = np.random.rand(Nf).astype(config.real(np))
    #         P2 = (3.0 * np.cos(theta)**2 - 1.0)
    #         cos_t = np.cos(t_fixed)
    #         Rmax_theta = 1.0 + self.a2 * P2 * cos_t
    #         r = np.cbrt(u3) * Rmax_theta
    #         x, y, z = self.sphere_to_cartesian(r, theta, phi)
    #         t_array = np.full(Nf, t_fixed)
    #         data = np.column_stack((x, y, z, t_array))
    #         all_data.append(data)
    #         print("---- DEBUG ----")
    #         print("x:", x.shape)
    #         print("y:", y.shape)
    #         print("z:", z.shape)
    #         print("theta:", theta.shape)
    #         print("phi:", phi.shape)
    #         print("----------------")
            
    #     GE_points_Ad = np.vstack(all_data)
    #     print(f"Total points: {GE_points_Ad.shape[0]}")
    #     print("\nGE_points_Ad shape:", GE_points_Ad.shape)
    #     np.savetxt('{:}/GE_points_Ad.tsv'.format(out_dir), GE_points_Ad, fmt='%16.8e', delimiter='\t', header='x\ty\tz\tt', comments='')
    #     return tf.convert_to_tensor(GE_points_Ad, config.real(tf))
            

    
    