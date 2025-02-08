"""
Author: Rafael Lemos - rafaellemos42@gmail.com
Date: 12/08/2024

License: MIT License
"""

from math import dist
from numpy import dot, arccos, degrees
from numpy.linalg import norm

from classes import Contact
from distances import distances
import conditions

import re

def contact_detection(protein):
    """
    Detects contacts between atoms in a given protein.

    Args:
        protein (Protein): The protein object containing chain, residue and atom objects.

    Returns:
        list: A list of Contact objects representing the detected contacts.
    """

    residues = list(protein.get_residues())
    contacts_nuc = []
    num_luana_res = 0
    num_luana_atom = 0
    num_puton = 0
    luana = set()
    luana_list = []
    puton = set()
    puton_list = []
    
    for i, residue1 in enumerate(residues[1:]):
        for _, residue2 in enumerate(residues[i+1:], start=i+1):
            
            if residue1.resnum == residue2.resnum and residue1.chain.id == residue2.chain.id: # ignores same residue
                continue
            
            if len(residue1.atoms) > 1 and len(residue2.atoms) > 1:
                ca1, ca2 = residue1.atoms[1], residue2.atoms[1] # alpha carbons
                distance_ca = dist((ca1.x, ca1.y, ca1.z), (ca2.x, ca2.y, ca2.z))
                
                # filter distant residues (static value then specific values)
                if distance_ca > 20.4:
                    continue
                else:
                    if residue1.nuc or residue2.nuc:
                        pass
                    else:
                        key = ''.join(sorted((residue1.resname, residue2.resname)))
                        if distance_ca > distances[key]:
                            continue

            # CHECKING FOR AROMATIC STACKINGS
            if residue1.ring and residue2.ring and residue1.nuc != residue2.nuc:
                ring1, ring2 = residue1.atoms[-1], residue2.atoms[-1] # RNG atoms
                distance = dist((ring1.x, ring1.y, ring1.z), (ring2.x, ring2.y, ring2.z))
                angle = calc_angle(residue1.normal_vector, residue2.normal_vector)
                if distance >= 2 and distance <= 5: # within aromatic stacking limits
                    if (160 <= angle < 180) or (0 <= angle < 20):
                        stack_type = "-parallel"
                    elif (80 <= angle < 100):
                        stack_type = "-perpendicular"
                    else:
                        stack_type = "-other"

                    contact_ring = Contact(protein.id, residue1.chain.id, residue1.resnum, residue1.resname, ring1.atomname, 
                                    protein.id, residue2.chain.id, residue2.resnum, residue2.resname, ring2.atomname, 
                                    float(f"{distance:.2f}"), "stacking"+stack_type, ring1, ring2)
                    
                    entry_luana = f"L = {residue1.chain.id}-{residue1.resnum}{residue1.resname} and {residue2.chain.id}-{residue2.resnum}{residue2.resname}"
                    if entry_luana not in luana:
                        num_luana_res += 1
                        luana.add(entry_luana)
                        luana_list.append(f"{entry_luana}")
                    num_luana_atom += 1
                    
                    contacts_nuc.append(contact_ring)
                    
            for atom1 in residue1.atoms:
                for atom2 in residue2.atoms:
                    
                    name1 = f"{atom1.residue.resname}:{atom1.atomname}" # matches the pattern from conditions dictionary
                    name2 = f"{atom2.residue.resname}:{atom2.atomname}"
                    
                    if residue1.nuc != residue2.nuc:
                        distance = dist((atom1.x, atom1.y, atom1.z), (atom2.x, atom2.y, atom2.z))
                        
                        # BLOCO DA LUANA:
                        if (name1 in conditions.contact_types and name2 in conditions.contact_types):

                            if distance <= 6: # max distance for contacts
                                
                                for contact_type, distance_range in conditions.categories.items():
                                    
                                    # if contact_type == 'hydrogen_bond' and (abs(residue2.resnum - residue1.resnum) <= 3): # skips alpha-helix for h-bonds
                                    #     continue
                                    if distance_range[0] <= distance <= distance_range[1]: # fits the range
                                        
                                        if conditions.contact_conditions[contact_type](name1, name2): # fits the type of contact                                       
                                            
                                            contact_nuc = Contact(protein.id, residue1.chain.id, residue1.resnum, residue1.resname, atom1.atomname, 
                                                            protein.id, residue2.chain.id, residue2.resnum, residue2.resname, atom2.atomname, 
                                                            float(f"{distance:.2f}"), contact_type, atom1, atom2)
                                                                                       
                                            entry_luana = f"L = {residue1.chain.id}-{residue1.resnum}{residue1.resname} and {residue2.chain.id}-{residue2.resnum}{residue2.resname}"
                                            
                                            if entry_luana not in luana:
                                                num_luana_res += 1
                                                luana.add(entry_luana)
                                                luana_list.append(f"{entry_luana}")
                                            num_luana_atom += 1
                                            contacts_nuc.append(contact_nuc)

                        # BLOCO DO PUTON:
                        if distance <= 3.5:
                            entry_puton = f"A = {residue1.chain.id}-{residue1.resnum}{residue1.resname} and {residue2.chain.id}-{residue2.resnum}{residue2.resname}"
                            if entry_puton not in puton:
                                num_puton += 1
                                puton.add(entry_puton)
                                puton_list.append(f"{entry_puton}")
                            continue
                                                                                    
    return contacts_nuc, num_luana_res, num_luana_atom, num_puton

def alphanumeric_key(s):
    # Split the string into parts, treating numbers as integers and keeping other parts as strings
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', s)]

def show_contacts(contacts):
    """
    Formats and summarizes contact information to be outputted to a file. Only works with the -o flag.

    Args:
        contacts (list): A list of Contact objects of a given protein.

    Returns:
        str: A formatted string summarizing the contact information.
    """
    
    output = []
    
    output.append("Chain1,Res1,ResName1,Atom1,Chain2,Res2,ResName2,Atom2,Distance,Type")
    for contact in contacts:
        output.append(contact.print_text())
        
    return "\n".join(output) # returns as a string to be written directly into the file


def calc_angle(vector1, vector2):
    """
    Calculates the angle between two ring vectors of aromatic residues

    Args:
        vector1 (tuple): The first vector (x, y, z).
        vector2 (tuple): The second vector (x, y, z).

    Returns:
        float: The angle between the vectors in degrees.
    """
    
    dot_product = dot(vector1, vector2)
    magnitude_product = norm(vector1) * norm(vector2) # normalizes the dot product
    angle = arccos(dot_product / magnitude_product) # angle in radians   
    
    return degrees(angle)
