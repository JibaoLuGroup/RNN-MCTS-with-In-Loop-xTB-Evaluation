import os
import time
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from chemtsv2.reward import Reward


class IPEA_reward(Reward):
    def get_objective_functions(conf):
        def IPEA_dip_gap(mol):
            output_dir = './result/IPEA_dip_gap_150w_200_26'
            os.makedirs(output_dir, exist_ok=True)

            XYZ_input = os.path.join(output_dir, f"InputMol{conf['gid']}.xyz")
            mol_wH = Chem.AddHs(mol)
            params = AllChem.ETKDG()
            AllChem.EmbedMultipleConfs(mol_wH, numConfs=30, params=params)
            Chem.MolToXYZFile(mol_wH, XYZ_input)

            if os.path.getsize(XYZ_input) == 0:
                print(f"Error: {XYZ_input} is empty ")
                return [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

            res = AllChem.MMFFOptimizeMoleculeConfs(mol_wH)
            min_energy = float('inf')
            min_idx = -1

            for i, (not_converged, energy) in enumerate(res):
                if not_converged == 0 and energy < min_energy:
                    min_energy = energy
                    min_idx = i

            if min_idx != -1:
                Chem.MolToXYZFile(mol_wH, XYZ_input, confId=min_idx)
                XYZ_input_opt = os.path.join(output_dir, f"InputMolopt_ok_{conf['gid']}.xyz")
            else:
                XYZ_input_opt = os.path.join(output_dir, f"InputMolopt_fail_{conf['gid']}.xyz")

            Chem.MolToXYZFile(mol_wH, XYZ_input_opt)

            current_dir = os.getcwd()
            os.chdir(output_dir)
            xtb_opt_log = f"xtb_opt_output_{conf['gid']}.log"
            xtb_opt_cmd = f"xtb {os.path.basename(XYZ_input_opt)} --opt > {xtb_opt_log}"
            os.system(xtb_opt_cmd)

            xtb_opt_xyz = "xtbopt.xyz"
            timeout = 30  
            interval = 1  
            elapsed_time = 0
            while not os.path.exists(xtb_opt_xyz) and elapsed_time < timeout:
                time.sleep(interval)
                elapsed_time += interval

            files_to_remove = ['xtbtopo.mol', 'xtbrestart', 'charges', 'wbo']
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)

            if os.path.exists(xtb_opt_xyz):
                new_filename = f"xtb_opt_{conf['gid']}.xyz"
                os.rename(xtb_opt_xyz, new_filename)
                xtb_vipea_log = f"xtb_vipea_output_{conf['gid']}.log"
                xtb_vipea_cmd = f"xtb {new_filename} --vipea > {xtb_vipea_log}"
                os.system(xtb_vipea_cmd)

                vipea_timeout = 30 
                vipea_elapsed_time = 0
                while not os.path.exists(xtb_vipea_log) and vipea_elapsed_time < vipea_timeout:
                    time.sleep(interval)
                    vipea_elapsed_time += interval

                files_to_remove = ['xtbtopo.mol', 'xtbrestart', 'charges', 'wbo']
                for file in files_to_remove:
                    if os.path.exists(file):
                        os.remove(file)

                xtb_dipole_log = f"xtb_dipole_output_{conf['gid']}.log"
                xtb_dipole_cmd = f"xtb {new_filename} --gfn2 --molden --dipole > {xtb_dipole_log}"
                os.system(xtb_dipole_cmd)

                stda_log = 'stda.txt'
                xtb4stda_log = 'xtb4stda_output.txt'
                xtb_stda_cmd = f"xtb4stda {new_filename} > {xtb4stda_log} && stda_v1.6.1.1 -xtb -e 10 > {stda_log}"
                os.system(xtb_stda_cmd)

            os.chdir(current_dir)

            ip_result = None
            ea_result = None
            dipole_result = None
            optical_gap_result = None
            homo_lumo_gap = None

            xtb_vipea_log = os.path.join(output_dir, f"xtb_vipea_output_{conf['gid']}.log")
            if os.path.exists(xtb_vipea_log):
                with open(xtb_vipea_log, 'r') as file:
                    for line in file:
                        if "HOMO-LUMO gap" in line:
                            homo_lumo_gap = float(line.split()[-3])
                        if "delta SCC IP (eV):" in line:
                            ip_result = float(line.split(":")[1].strip())
                        if "delta SCC EA (eV):" in line:
                            ea_result = float(line.split(":")[1].strip())
                            break

            molecular_dipole_found = False
            xtb_dipole_log = os.path.join(output_dir, f"xtb_dipole_output_{conf['gid']}.log")
            if os.path.exists(xtb_dipole_log):
                with open(xtb_dipole_log, 'r') as file:
                    for line in file:
                        if "molecular dipole:" in line:
                            molecular_dipole_found = True 
                        elif molecular_dipole_found and "full:" in line:
                            dipole_result = float(line.split()[-1])  
                            break  

            stda_log = os.path.join(output_dir, f"stda.txt")
            if os.path.exists(stda_log):
                with open(stda_log, 'r') as file:
                    for line in file:
                        if "state" in line and "eV" in line:
                            next_line = file.readline().strip()
                            columns = next_line.split()
                            optical_gap_result = float(columns[1])
                            break  
            if ip_result is None or ea_result is None or dipole_result is None or optical_gap_result is None:
                return [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

            return [homo_lumo_gap, ip_result, ea_result, dipole_result, optical_gap_result]

        return [IPEA_dip_gap]

    def calc_reward_from_objective_values(values, conf):
        HLGap, IP, EA, dipole, optical_gap = values[0]
        if any(map(lambda x: x != x, [HLGap, IP, EA, dipole, optical_gap])):  # Check for NaN
            return -1

        IP_normalized = 1 / (1 + math.exp(-0.3*(IP - 8)))  # 对应f(x) = 1 / (1 + e^{-(x - 7)})
        EA_normalized = 1 / (1 + math.exp(-0.3*(EA - 2)))  # 对应f(x) = 1 / (1 + e^{-(x - 1.5)})

        reward = (0.5 * IP_normalized) + (0.5 * EA_normalized) 

        return reward
