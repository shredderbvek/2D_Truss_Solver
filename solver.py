import numpy as np
import matplotlib.pyplot as plt

def draw_line(x1, y1, x2, y2, ls, c):
    plt.plot([x1, x2], [y1, y2], linestyle = ls, color=c)

# Define the file path
file_path = input("Input the data file in .txt format:")
#-------------------------------------------------------------------------------------------------------------
# READ DATA FROM THE MESH FILES
#-------------------------------------------------------------------------------------------------------------
# Initializes dictionaries for structured data
elements = []  # List to hold each element's data
nodes = []  # List to hold each node's data
prescribed_displacements = []  # List to hold Dirichlet boundary conditions

# Reads and parses the file
with open(file_path, 'r') as file:
    lines = file.readlines()
    current_section = None

    for line in lines:
        line = line.strip()
        
        # Identifies sections
        if line.startswith("ELEMENTS"):
            current_section = "ELEMENTS"
        elif line.startswith("NO_OF_NODES"):
            current_section = "NO_OF_NODES"
        elif line.startswith("NODES_WITH_PRESCRIBED_DISPLACEMENT"):
            current_section = "NODES_WITH_PRESCRIBED_DISPLACEMENT"
        elif line and not line.startswith("="):  # Non-empty, non-header lines
            data = list(map(float, line.split()))
            
            # Parses according to the section
            if current_section == "ELEMENTS":
                element_data = {
                    "element_number": int(data[0]),
                    "global_node_1": int(data[1]),
                    "global_node_2": int(data[2]),
                    "modulus_of_elasticity": data[3],
                    "area": data[4]
                }
                elements.append(element_data)
            elif current_section == "NO_OF_NODES":
                node_data = {
                    "node": int(data[0]),
                    "x_coordinate": data[1],
                    "y_coordinate": data[2],
                    "x_external_force": data[3],
                    "y_external_force": data[4]
                }
                nodes.append(node_data)
            elif current_section == "NODES_WITH_PRESCRIBED_DISPLACEMENT":
                displacement_data = {
                    "node": int(data[0]),
                    "x_displacement": data[1],
                    "y_displacement": data[2]
                }
                prescribed_displacements.append(displacement_data)

# Prints the parsed data for verification
print("Elements:")
for element in elements:
    print(element)

print("\nNodes:")
for node in nodes:
    print(node)

print("\nPrescribed Displacements:")
for displacement in prescribed_displacements:
    print(displacement)
#extracting elemental data   
elemental_data = np.array([
    [element["element_number"], element["global_node_1"], element["global_node_2"], element["modulus_of_elasticity"], element["area"]]
    for element in elements
])
#extracting particular elemental data
no_of_elements = len(elements)
element_order = elemental_data[:, 0].astype(int)
connectivity_mat = elemental_data[:, 1: 3].astype(int)
elasticity_mat = elemental_data[:, 3]
area_mat = elemental_data[:, 4]
#extracting nodal data
nodal_data = np.array(
    [
        [item["node"], item["x_coordinate"], item["y_coordinate"], item["x_external_force"], item["y_external_force"]]
        for item in nodes
    ]
)
#extracting particular elemental data
no_of_nodes = len(nodes)
node_order = nodal_data[:,0].astype(int)
nodal_coord = nodal_data[:, 1:3]
nodal_force = nodal_data[:, 3:]
nodal_coord = nodal_coord[node_order - 1, :]
nodal_force = nodal_force[node_order - 1, :]
x_1 = nodal_coord[connectivity_mat[:,0]-1,0]
x_2 = nodal_coord[connectivity_mat[:,1]-1,0]
y_1 = nodal_coord[connectivity_mat[:,0]-1,1]
y_2 = nodal_coord[connectivity_mat[:,1]-1,1]

#extracting dirichlet boundary condition data
dirichlet_data = np.array(
    [
        [item["node"], item["x_displacement"], item["y_displacement"]]
        for item in prescribed_displacements
    ]
)
#extracting particular dirichlet data
no_of_nodes_dirichlet = len(prescribed_displacements)
nodes_dirichlet = dirichlet_data[:, 0].astype(int)
values_dirichlet = dirichlet_data[:, 1:]
#Creates a vector of lentgh nOfNodes and set 1 if the node has a imposed
#displacement or 0 otherwise
is_node_fixed = np.zeros((no_of_nodes, 1))
is_node_fixed[nodes_dirichlet-1] = np.ones((no_of_nodes_dirichlet, 1))

no_of_free_nodes = no_of_nodes - no_of_nodes_dirichlet
free_nodes = np.ones((no_of_free_nodes, 1))
counter_free_nodes = 0

for i in range(no_of_nodes):
    if is_node_fixed[i] == 0:
        counter_free_nodes = counter_free_nodes + 1
        free_nodes[counter_free_nodes-1] = i + 1
#----------------------------------------------------------------------------------------------------------------------------
# Computation of stiffness matrix and field variable (displacements)
#----------------------------------------------------------------------------------------------------------------------------
elemental_length = np.zeros((no_of_elements, 1))
elemental_angle = np.zeros((no_of_elements, 1))
for i in range(len(elemental_length)):
    elemental_length[i] = np.sqrt((x_2[i] - x_1[i])**2 + (y_2[i] - y_1[i])**2)
    elemental_angle[i] = np.degrees(np.arctan((y_2[i] - y_1[i])/(x_2[i] - x_1[i])))
    

# Initialize global stiffness matrix
global_K = np.zeros((no_of_nodes * 2, no_of_nodes * 2))

for i_element in range(no_of_elements):
    # Compute cosine and sine of the angle
    C = np.cos(np.radians(elemental_angle[i_element]))
    S = np.sin(np.radians(elemental_angle[i_element]))

    # Transformation matrix for the element
    T_matrix = np.array([
        [C**2, C*S, -C**2, -C*S],
        [C*S, S**2, -C*S, -S**2],
        [-C**2, -C*S, C**2, C*S],
        [-C*S, -S**2, C*S, S**2]
    ])

    # Local stiffness matrix for the element
    elemental_K = (elasticity_mat[i_element] * area_mat[i_element] / elemental_length[i_element]) * T_matrix

    # Map global nodes and expand DOFs (2 DOFs per node)
    global_element_nodes = connectivity_mat[i_element, :]
    global_dofs = np.ravel([[2*node-2, 2*node-1] for node in global_element_nodes])

    # Update the global stiffness matrix
    for i in range(len(global_dofs)):
        for j in range(len(global_dofs)):
            global_K[global_dofs[i], global_dofs[j]] += elemental_K[i, j]

    
global_F = np.zeros((no_of_nodes*2, 1))
for i_node in range(no_of_nodes):
    global_F[i_node*2] = nodal_force[i_node, 0]
    global_F[i_node*2 + 1] = nodal_force[i_node, 1]

global_displacement = np.ones((no_of_nodes*2, 1))


#global_dofs = np.ravel([[2*node-2, 2*node-1] for node in nodes_dirichlet])
counter_x = 0
for node in nodes_dirichlet:
    global_dofs = np.ravel([2*node-2, 2*node-1])
    counter_y = 0
    for dof in global_dofs:
        global_displacement[dof] = values_dirichlet[counter_x, counter_y]
        counter_y = counter_y + 1
    counter_x = counter_x + 1

# Identify DOFs associated with prescribed displacements
prescribed_dofs = []
for node in nodes_dirichlet:
    prescribed_dofs.extend([2 * node - 2, 2 * node - 1])  # x and y DOFs for the node

# Identify free DOFs
all_dofs = np.arange(global_K.shape[0])
free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

# Extract reduced stiffness matrix (K_reduced) and reduced force vector (F_reduced)
K_reduced = global_K[np.ix_(free_dofs, free_dofs)]
F_reduced = global_F[free_dofs]

# Solve for displacements of free DOFs
U_free = np.linalg.solve(K_reduced, F_reduced)

# Update the global displacement vector with solved free displacements
for i, dof in enumerate(free_dofs):
    global_displacement[dof] = U_free[i]

# Print the results
print("\nReduced Stiffness Matrix (K_reduced):")
print(K_reduced)

print("\nReduced Force Vector (F_reduced):")
print(F_reduced)

print("\n------------------------------NODAL DISPLACEMENTS----------------------------------------------\n")
for node in node_order:
    print("node:", node, "x_disp:", global_displacement[2*node - 2], "y_disp:", global_displacement[2*node - 1])
#--------------------------------------------------------------------------------------------------------------
#post processing
#--------------------------------------------------------------------------------------------------------------
elemental_stress = np.zeros((no_of_elements, 1))  # Initialize stress array

for i in range(no_of_elements):
    # Cosine and sine of the element's angle
    C = np.cos(np.radians(elemental_angle[i]))
    S = np.sin(np.radians(elemental_angle[i]))
    
    # Derivative of displacement shape function (strain-displacement relation)
    C_dash = (elasticity_mat[i] / elemental_length[i]) * np.array([-C, -S, C, S])
    C_dash = C_dash.flatten()
    
    # DOFs associated with the current element
    dofs = np.ravel([
        2 * connectivity_mat[i, 0] - 2,  # DOF for x of node 1
        2 * connectivity_mat[i, 0] - 1,  # DOF for y of node 1
        2 * connectivity_mat[i, 1] - 2,  # DOF for x of node 2
        2 * connectivity_mat[i, 1] - 1   # DOF for y of node 2
    ]).astype(int)
    
    # Nodal displacement vector for the element (flattened to 1D)
    nodal_displacement_mat = global_displacement[dofs].flatten()
    
    # Compute stress using strain-displacement relation
    elemental_stress[i] = np.dot(C_dash, nodal_displacement_mat)

# Print the elemental stresses
print("\n---------------------------------ELEMENTAL STRESS-------------------------------------------------\n")
for element in element_order:
    print("Element:", element, "Stress:", elemental_stress[element - 1] )
#undeforemed view
for node_pair in connectivity_mat:
    x1, y1 = nodal_coord[node_pair[0]-1]
    x2, y2 = nodal_coord[node_pair[1]-1]
    draw_line(x1, y1, x2, y2, "--", "c")
# Deformed view
scale = 17311  # Scale factor for visualizing the deformation
for node_pair in connectivity_mat:
    # Original coordinates of the two nodes
    x1, y1 = nodal_coord[node_pair[0] - 1]
    x2, y2 = nodal_coord[node_pair[1] - 1]
    
    # Displacements for the two nodes
    u1_x = global_displacement[2 * (node_pair[0] - 1)][0]
    u1_y = global_displacement[2 * (node_pair[0] - 1) + 1][0]
    u2_x = global_displacement[2 * (node_pair[1] - 1)][0]
    u2_y = global_displacement[2 * (node_pair[1] - 1) + 1][0]
    
    # Deformed coordinates
    x1_def = x1 + scale * u1_x
    y1_def = y1 + scale * u1_y
    x2_def = x2 + scale * u2_x
    y2_def = y2 + scale * u2_y
    
    # Plot the deformed element
    draw_line(x1_def, y1_def, x2_def, y2_def, "-", "r")
# Set axis equal for proper proportions
plt.axis('equal')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Nodal Displacement")
plt.show()
