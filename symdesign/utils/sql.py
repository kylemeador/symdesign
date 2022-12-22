from collections import OrderedDict
from typing import AnyStr

from mysql.connector import MySQLConnection, Error
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, create_engine
from sqlalchemy.orm import declarative_base, relationship
# from sqlalchemy.orm import Mapped, mapped_column, Session


Base = declarative_base()


class Designs(Base):
    __tablename__ = 'designs'

    id = Column(Integer, primary_key=True)
    pose = Column(String, nullable=False)  # String(60)
    design = Column(String, nullable=False)  # String(60)
    # residues_id = relationship('residues', back_populates='')  # LIST

    # Fragment features
    nanohedra_score_normalized = Column(Float)  # , nullable=False)
    nanohedra_score_center_normalized = Column(Float)  # , nullable=False)
    nanohedra_score = Column(Float)  # , nullable=False)
    nanohedra_score_center = Column(Float)  # , nullable=False)
    number_fragment_residues_total = Column(Integer)  # , nullable=False)
    number_fragment_residues_center = Column(Integer)  # , nullable=False)
    multiple_fragment_ratio = Column(Float)  # , nullable=False)
    percent_fragment_helix = Column(Float)  # , nullable=False)
    percent_fragment_strand = Column(Float)  # , nullable=False)
    percent_fragment_coil = Column(Float)  # , nullable=False)
    number_of_fragments = Column(Integer)  # , nullable=False)
    percent_residues_fragment_interface_total = Column(Float)  # , nullable=False)
    percent_residues_fragment_interface_center = Column(Float)  # , nullable=False)
    # Pose features
    number_interface_residues = Column(Integer)  # , nullable=False)
    number_design_residues = Column(Integer)  # , nullable=False)
    pose_length = Column(Integer)  # , nullable=False)
    pose_thermophilicity = Column(Float)  # , nullable=False)
    minimum_radius = Column(Float)  # , nullable=False)
    maximum_radius = Column(Float)  # , nullable=False)
    interface_b_factor_per_residue = Column(Float)  # , nullable=False)
    interface_secondary_structure_fragment_topology = Column(String(120))  # , nullable=False)
    interface_secondary_structure_fragment_count = Column(Integer)  # , nullable=False)
    interface_secondary_structure_topology = Column(String(120))  # , nullable=False)
    interface_secondary_structure_count = Column(Integer)  # , nullable=False)
    design_dimension = Column(Integer)  # , nullable=False)
    entity_max_radius = Column(Float)  # Has a # after entity LIST
    entity_min_radius = Column(Float)  # Has a # after entity LIST
    entity_name = Column(String(30))  # Has a # after entity LIST
    entity_number_of_residues = Column(Integer)  # Has a # after entity LIST
    entity_radius = Column(Float)  # Has a # after entity LIST
    entity_symmetry_group = Column(String(4))  # Has a # after entity LIST
    entity_n_terminal_helix = Column(Boolean)  # Has a # after entity LIST
    entity_c_terminal_helix = Column(Boolean)  # Has a # after entity LIST
    entity_n_terminal_orientation = Column(Boolean)  # Has a # after entity LIST
    entity_c_terminal_orientation = Column(Boolean)  # Has a # after entity LIST
    entity_thermophile = Column(Boolean)  # Has a # after entity LIST
    entity_interface_secondary_structure_fragment_topology = Column(String(60))  # Has a # after entity LIST
    entity_interface_secondary_structure_topology = Column(String(60))  # Has a # after entity LIST
    entity_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    entity_min_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    entity_max_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    entity_number_of_residues_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    entity_radius_average_deviation = Column(Float)  # , nullable=False)
    entity_min_radius_average_deviation = Column(Float)  # , nullable=False)
    entity_max_radius_average_deviation = Column(Float)  # , nullable=False)
    entity_number_of_residues_average_deviation = Column(Float)  # , nullable=False)
    # Rosetta metrics
    buns_complex = Column(Integer)  # , nullable=False)
    buns_unbound = Column(Integer)  # Has a # after buns LIST
    buried_unsatisfied_hbonds = Column(Integer)  # , nullable=False)
    buried_unsatisfied_hbond_density = Column(Float)  # , nullable=False)
    contact_count = Column(Integer)  # , nullable=False)
    entity_interface_connectivity = Column(Float)  # Has a # after entity LIST
    favor_residue_energy = Column(Float)  # , nullable=False)
    interaction_energy_complex = Column(Float)  # , nullable=False)
    interface_energy_density = Column(Float)  # , nullable=False)
    interface_bound_activation_energy = Column(Float)  # , nullable=False)
    interface_solvation_energy = Column(Float)  # , nullable=False)
    interface_solvation_energy_activation = Column(Float)  # , nullable=False)
    interaction_energy_per_residue = Column(Float)  # , nullable=False)
    interface_separation = Column(Float)  # , nullable=False)
    rmsd_complex = Column(Float)  # , nullable=False)
    rosetta_reference_energy = Column(Float)  # , nullable=False)
    shape_complementarity = Column(Float)  # , nullable=False)
    # solvation_energy = Column(Float)  # , nullable=False)
    # solvation_energy_complex = Column(Float)  # , nullable=False)
    # Sequence metrics
    percent_mutations = Column(Float)  # , nullable=False)
    number_of_mutations = Column(Integer)  # , nullable=False)
    entity_percent_mutations = Column(Float)  # Has a # after entity LIST
    entity_number_of_mutations = Column(Integer)  # Has a # after entity LIST
    # SymDesign metrics
    interface_local_density = Column(Float)  # , nullable=False)
    interface_composition_similarity = Column(Float)  # , nullable=False)
    collapse_significance_by_contact_order_z_mean = Column(Float)  # , nullable=False)
    collapse_increased_z_mean = Column(Float)  # , nullable=False)
    collapse_variance = Column(Float)  # , nullable=False)
    collapse_sequential_peaks_z_mean = Column(Float)  # , nullable=False)
    collapse_sequential_z_mean = Column(Float)  # , nullable=False)
    interface_area_to_residue_surface_ratio = Column(Float)  # , nullable=False)
    sequence = Column(String(10000))  # , nullable=False)
    # Summed Residues metrics
    # -----------------------
    # column name is changed from Rosetta energy values
    interface_energy_complex = Column(Float)  # , nullable=False)
    interface_energy_bound = Column(Float)  # , nullable=False)
    interface_energy_unbound = Column(Float)  # , nullable=False)
    interface_energy = Column(Float)  # , nullable=False)
    interface_solvation_energy_complex = Column(Float)  # , nullable=False)
    interface_solvation_energy_bound = Column(Float)  # , nullable=False)
    interface_solvation_energy_unbound = Column(Float)  # , nullable=False)
    number_of_hbonds = Column(Integer)  # , nullable=False)
    # column name is changed
    area_hydrophobic_complex = Column(Float)  # , nullable=False)
    area_hydrophobic_unbound = Column(Float)  # , nullable=False)
    area_polar_complex = Column(Float)  # , nullable=False)
    area_polar_unbound = Column(Float)  # , nullable=False)
    area_total_complex = Column(Float)  # , nullable=False)
    area_total_unbound = Column(Float)  # , nullable=False)
    interface_area_polar = Column(Float)  # , nullable=False)
    interface_area_hydrophobic = Column(Float)  # , nullable=False)
    interface_area_total = Column(Float)  # , nullable=False)
    # Done column name is changed
    coordinate_constraint = Column(Float)  # , nullable=False)
    residue_favored = Column(Float)  # , nullable=False)
    # SymDesign measurements
    contact_order = Column(Float)  # , nullable=False)
    errat_deviation = Column(Float)  # , nullable=False)
    sasa_hydrophobic_complex = Column(Float)  # , nullable=False)
    sasa_polar_complex = Column(Float)  # , nullable=False)
    # NOT taken sasa_relative_complex = Column(Float)  # , nullable=False)
    sasa_hydrophobic_bound = Column(Float)  # , nullable=False)
    sasa_polar_bound = Column(Float)  # , nullable=False)
    # NOT taken sasa_relative_bound = Column(Float)  # , nullable=False)
    interior = Column(Float)  # , nullable=False)
    surface = Column(Float)  # , nullable=False)
    support = Column(Float)  # , nullable=False)
    rim = Column(Float)  # , nullable=False)
    core = Column(Float)  # , nullable=False)
    percent_interface_area_hydrophobic = Column(Float)  # , nullable=False)
    percent_interface_area_polar = Column(Float)  # , nullable=False)
    percent_core = Column(Float)  # , nullable=False)
    percent_rim = Column(Float)  # , nullable=False)
    percent_support = Column(Float)  # , nullable=False)
    # Collapse measurements
    # NOT taken hydrophobic_collapse = Column(Float)  # , nullable=False)
    collapse_deviation_magnitude = Column(Float)  # , nullable=False)
    collapse_increase_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    collapse_increased_z = Column(Float)  # , nullable=False)
    collapse_new_positions = Column(Boolean)  # , nullable=False)
    collapse_new_position_significance = Column(Float)  # , nullable=False)
    collapse_sequential_peaks_z = Column(Float)  # , nullable=False)
    collapse_sequential_z = Column(Float)  # , nullable=False)
    collapse_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    # ProteinMPNN score terms
    proteinmpnn_loss_complex = Column(Float)  # , nullable=False)
    proteinmpnn_loss_unbound = Column(Float)  # , nullable=False)
    sequence_loss_design = Column(Float)  # , nullable=False)
    sequence_loss_design_per_residue = Column(Float)  # , nullable=False)
    sequence_loss_evolution = Column(Float)  # , nullable=False)
    sequence_loss_evolution_per_residue = Column(Float)  # , nullable=False)
    sequence_loss_fragment = Column(Float)  # , nullable=False)
    sequence_loss_fragment_per_residue = Column(Float)  # , nullable=False)
    proteinmpnn_v_design_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_evolution_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_fragment_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    dock_collapse_significance_by_contact_order_z_mean = Column(Float)  # , nullable=False)
    dock_collapse_increased_z_mean = Column(Float)  # , nullable=False)
    dock_collapse_variance = Column(Float)  # , nullable=False)
    dock_collapse_sequential_peaks_z_mean = Column(Float)  # , nullable=False)
    dock_collapse_sequential_z_mean = Column(Float)  # , nullable=False)
    dock_collapse_deviation_magnitude = Column(Float)  # , nullable=False)
    dock_collapse_increase_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    dock_collapse_increased_z = Column(Float)  # , nullable=False)
    dock_collapse_new_positions = Column(Boolean)  # , nullable=False)
    dock_collapse_new_position_significance = Column(Float)  # , nullable=False)
    dock_collapse_sequential_peaks_z = Column(Float)  # , nullable=False)
    dock_collapse_sequential_z = Column(Float)  # , nullable=False)
    dock_collapse_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    # Observed in profile measurements
    observed_design = Column(Boolean)  # , nullable=False)
    observed_evolution = Column(Boolean)  # , nullable=False)
    observed_fragment = Column(Boolean)  # , nullable=False)
    observed_interface = Column(Boolean)  # , nullable=False)
    # Direct coupling analysis energy
    dca_energy = Column(Boolean)  # , nullable=False)
    # -----------------------

    # protocol = relationship()  # LIST
    # addresses = relationship(
    #     "Address", back_populates="user", cascade="all, delete-orphan"
    # )

    # def __repr__(self):
    #     return f"Trajectory(id={self.id!r}, pose={self.pose!r}, name={self.name!r})"


class Residues(Base):
    __tablename__ = 'residues'

    id = Column(Integer, primary_key=True)
    design_id = Column(Integer, ForeignKey('designs.id'))  # , nullable=False)
    # Residue position (surrogate for residue number) and type information
    pose = Column(String, nullable=False)  # String(60)
    design = Column(String, nullable=False)  # String(60)
    index = Column(Integer, nullable=False)
    type = Column(String(1), nullable=False)
    design_residue = Column(Boolean)
    # Rosetta energy values
    complex = Column(Float)  # , nullable=False)
    bound = Column(Float)  # , nullable=False)
    unbound = Column(Float)  # , nullable=False)
    energy_delta = Column(Float)  # , nullable=False)
    solv_complex = Column(Float)  # , nullable=False)
    solv_bound = Column(Float)  # , nullable=False)
    solv_unbound = Column(Float)  # , nullable=False)
    hbond = Column(Boolean)  # , nullable=False)
    coordinate_constraint = Column(Float)  # , nullable=False)
    residue_favored = Column(Float)  # , nullable=False)
    # SymDesign measurements
    contact_order = Column(Float)  # , nullable=False)
    errat_deviation = Column(Float)  # , nullable=False)
    sasa_hydrophobic_complex = Column(Float)  # , nullable=False)
    sasa_polar_complex = Column(Float)  # , nullable=False)
    sasa_relative_complex = Column(Float)  # , nullable=False)
    sasa_hydrophobic_bound = Column(Float)  # , nullable=False)
    sasa_polar_bound = Column(Float)  # , nullable=False)
    sasa_relative_bound = Column(Float)  # , nullable=False)
    bsa_hydrophobic = Column(Float)  # , nullable=False)
    bsa_polar = Column(Float)  # , nullable=False)
    bsa_total = Column(Float)  # , nullable=False)
    sasa_total_bound = Column(Float)  # , nullable=False)
    sasa_total_complex = Column(Float)  # , nullable=False)
    interior = Column(Float)  # , nullable=False)
    surface = Column(Float)  # , nullable=False)
    support = Column(Float)  # , nullable=False)
    rim = Column(Float)  # , nullable=False)
    core = Column(Float)  # , nullable=False)
    # Collapse measurements
    hydrophobic_collapse = Column(Float)  # , nullable=False)
    collapse_deviation_magnitude = Column(Float)  # , nullable=False)
    collapse_increase_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    collapse_increased_z = Column(Float)  # , nullable=False)
    collapse_new_positions = Column(Boolean)  # , nullable=False)
    collapse_new_position_significance = Column(Float)  # , nullable=False)
    collapse_sequential_peaks_z = Column(Float)  # , nullable=False)
    collapse_sequential_z = Column(Float)  # , nullable=False)
    collapse_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    # ProteinMPNN score terms
    proteinmpnn_loss_complex = Column(Float)  # , nullable=False)
    proteinmpnn_loss_unbound = Column(Float)  # , nullable=False)
    sequence_loss_design = Column(Float)  # , nullable=False)
    sequence_loss_evolution = Column(Float)  # , nullable=False)
    sequence_loss_fragment = Column(Float)  # , nullable=False)
    proteinmpnn_v_design_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_evolution_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_fragment_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    dock_collapse_deviation_magnitude = Column(Float)  # , nullable=False)
    dock_collapse_increase_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    dock_collapse_increased_z = Column(Float)  # , nullable=False)
    dock_collapse_new_positions = Column(Boolean)  # , nullable=False)
    dock_collapse_new_position_significance = Column(Float)  # , nullable=False)
    dock_collapse_sequential_peaks_z = Column(Float)  # , nullable=False)
    dock_collapse_sequential_z = Column(Float)  # , nullable=False)
    dock_collapse_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    # Observed in profile measurements
    observed_design = Column(Boolean)  # , nullable=False)
    observed_evolution = Column(Boolean)  # , nullable=False)
    observed_fragment = Column(Boolean)  # , nullable=False)
    observed_interface = Column(Boolean)  # , nullable=False)
    # Direct coupling analysis energy
    dca_energy = Column(Boolean)  # , nullable=False)

    # user = relationship("User", back_populates="addresses")

    # def __repr__(self):
    #     return f"Residues(id={self.id!r}, index={self.index!r})"


def start_db(db: AnyStr):
    """"""
    # engine = create_engine('sqlite+pysqlite:///:memory:', echo=True, future=True)
    engine = create_engine(f'sqlite:///{db}', echo=True, future=True)
    Base.metadata.create_all(engine)

    return engine


# db = '/absolute/path/to/foo.db'
# engine = start_db(db)
#
# with Session(engine) as session:
#
#     session.add_all([spongebob, sandy, patrick])
#
#     session.commit()
#
#
# # Example of starting a Session and changing data in a table
# session = Session(engine)
# stmt = select(Residues).where(Residues.index.in_(["spongebob", "sandy"]))
#
# for residue in session.scalars(stmt):
#     print(residue)
#     residue.complex = 3.1
#
# session.commit()


class Mysql:

    __instance   = None
    __host       = None
    __user       = None
    __password   = None
    __database   = None
    __session    = None
    __connection = None
    __dictionary = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance or not cls.__database:
            cls.__instance = super(Mysql, cls).__new__(cls)  # , *args, **kwargs)
        return cls.__instance

    def __init__(self, host='localhost', user='root', password='', database=''):
        self.__host     = host
        self.__user     = user
        self.__password = password
        self.__database = database

    def __open(self):
        cnx = None
        try:
            cnx = MySQLConnection(host=self.__host, user=self.__user, password=self.__password, database=self.__database)
            self.__connection = cnx
            self.__session    = cnx.cursor(dictionary=self.__dictionary)
        except Error as e:
            print('Error %d: %s' % (e.args[0], e.args[1]))

    def __close(self):
        self.__session.close()
        self.__connection.close()
        
    def dictionary_on(self):
        self.__dictionary = True
        
    def dictionary_off(self):
        self.__dictionary = False
        
    def concatenate_table_columns(self, source, columns):
        table_columns = []
        for column in columns:
            table_columns.append(source + '.' + column)
        return tuple(table_columns)

    def select(self, table, where=None, *args, **kwargs):
        result = None
        query = 'SELECT '
        if not args:
            print('TRUE')
            keys = ['*']
        else:
            keys = args
        print(keys)
        values = tuple(kwargs.values())
        l = len(keys) - 1

        for i, key in enumerate(keys):
            query += key
            if i < l:
                query += ","

        query += ' FROM %s' % table

        if where:
            query += " WHERE %s" % where

        print(query)
        self.__open()
        self.__session.execute(query, values)
#         number_rows = self.__session.rowcount
#         number_columns = len(self.__session.description)
#         print(number_rows, number_columns)
#         if number_rows >= 1 and number_columns > 1:
        result = [item for item in self.__session.fetchall()]
#         else:
#             print('if only')
#             result = [item[0] for item in self.__session.fetchall()]
        self.__close()

        return result
    
    def join(self, table1, table2, where=None, join_type='inner'):
        """SELECT
            m.member_id, 
            m.name member, 
            c.committee_id, 
            c.name committee
        FROM
            members m
        INNER JOIN committees c USING(name);"""

    def insert_into(self, table, columns, source=None, where=None, equivalents=None, *args, **kwargs):           
        query = "INSERT INTO %s SET " % table
        table_columns = self.concatenate_table_columns(table, columns)
        
        if source:
            column_equivalents = self.concatenate_table_columns(source, columns)
            # column_and_equivalents = [(columns[i], column_equivalents[i]) for i in range(len(columns))]
            pre_query_columns = [column + ' = %s' for column in table_columns]
            pre_query_columns = [pre_query_columns[i] % column_equivalents[i] for i in range(len(column_equivalents))]
            # print(pre_query_columns)
            query += ", ".join(pre_query_columns) # + ") VALUES (" + ", ".join(["%s"] * len(columns)) + ")"
        
        if kwargs:
            keys   = kwargs.keys()
            values = tuple(kwargs.values()) + tuple(args)
            l = len(keys) - 1
            for i, key in enumerate(keys):
                query += key+ " = %s"
                if i < l:
                    query += ","
                ## End if i less than 1
            ## End for keys
        if where:
            query += " WHERE %s" % where
            
#         print(query)
#         stop = columns[30]
        self.__open()
        self.__session.execute(query) #, values)
        self.__connection.commit()

        # Obtain rows affected
        update_rows = self.__session.rowcount
        self.__close()

        return update_rows

    def insert(self, table, columns, values, *args, **kwargs):
        # values = None
        query = "INSERT INTO %s " % table
        query += "(" + ",".join(["%s"] * len(columns)) % tuple(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")"

        if kwargs:
            keys = kwargs.keys()
            values = tuple(kwargs.values())
            query += "(" + ",".join(["%s"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(["%s"] * len(values)) + ")"
        elif args:
            values = args
            query += " VALUES(" + ",".join(["%s"] * len(values[0][0])) + ")"
        
#         print(query)
#         print(values)

        self.__open()
        self.__session.execute(query, values)
        self.__connection.commit()
        self.__close()
        
        return self.__session.lastrowid
    
    def insert_multiple(self, table, columns, values, *args, **kwargs):
        # values = None
        query = "INSERT INTO %s " % table
        query += "(" + ",".join(["`%s`"] * len(columns)) % tuple(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")"
        
        if kwargs:
            keys = kwargs.keys()
            values = tuple(kwargs.values())
            query += "(" + ",".join(["`%s`"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(["%s"] * len(values)) + ")"
        elif args:
            query += "(" + ",".join(["`%s`"] * len(columns)) % tuple(columns) + ")"
            query += " VALUES(" + ",".join(["%s"] * len(args)) + ")"
        
        # print(query)
        # print(values[0][0])
        self.__open()
        self.__session.executemany(query, values)
        self.__connection.commit()
        self.__close()
        
        return self.__session.lastrowid

    def delete(self, table, where=None, *args):
        query = "DELETE FROM %s" % table
        if where:
            query += ' WHERE %s' % where

        values = tuple(args)

        self.__open()
        self.__session.execute(query, values)
        self.__connection.commit()

        # Obtain rows affected
        delete_rows = self.__session.rowcount
        self.__close()

        return delete_rows

    def select_advanced(self, sql, *args):
        od = OrderedDict(args)
        query  = sql
        values = tuple(od.values())
        self.__open()
        self.__session.execute(query, values)
        number_rows = self.__session.rowcount
        number_columns = len(self.__session.description)

        if number_rows >= 1 and number_columns > 1:
            result = [item for item in self.__session.fetchall()]
        else:
            result = [item[0] for item in self.__session.fetchall()]

        self.__close()
        
        return result
