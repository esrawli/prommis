#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################

#################################################################################
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

r"""

Ion Exchange Parameter Estimation Model
=======================================

Author: E. Soraya Rawlings

This example uses the ion exchange multicomponent (IXMC) model and
parmest package to estimate three key parameters from breakthrough
data under the Freundlich isotherm assumption:

#. :math:`mass\_transfer\_coeff`: Overall mass transfer coefficient
#. :math:`freundlich\_n`       : Freundlich exponent n
#. :math:`bv\_50`              : Bed volumes required for 50% breakthrough

The parmest model reads a CSV file that contains one or more
breakthrough curves, selects a single curve, constructs an IX model,
generates a list of ``IXExperiment`` objects for each data point, and
runs a scaled non‑linear least‑squares fit to obtain the parameters.

Configuration Arguments
-----------------------
The ``IXParmest`` class requires the following keyword arguments
(passed through the ``main`` helper):

.. csv-table:: Required keyword arguments in ``IXParmest`` class.
   :header: "Argument", "Type", "Description"

   ":math:`curve\_file`", ":math:`str`", "CSV file with breakthrough data."
   ":math:`data\_name`", ":math:`str`", "Base name of the CSV data file."
   ":math:`curve`", ":math:`int`", "Identifier number of the curve to analyse in breakthrough data file."
   ":math:`num\_traps`", ":math:`int`", "Number of trapezoids for the numerical integration in the IX unit. "
   ":math:`c\_trap\_min`", ":math:`float`", "Minimum concentration for trapezoidal rule."
   ":math:`resin\_data`", ":math:`dict`", "Resin properties for the IX unit."
   ":math:`initial\_guess\_dict`", ":math:`dict`", "Initial values for the parameters to be estimated."
   ":math:`set\_bounds`", ":math:`dict`", "Bounds for the IX unit variables, including the parameters to be estimated."
   ":math:`component\_data`", ":math:`dict`", "Properties of target ion."
   ":math:`regenerant`", ":math:`str`", "Regenerant name for the IX unit."
   ":math:`fix\_initial\_guess`", ":math:`dict`", "Values that are fixed before estimation."
   ":math:`scale\_from\_value`", ":math:`list[str]`", "Variables whose scaling factor is 1/value."
   ":math:`diff\_calculation`", ":math:`str`", "Method for estimating diffusivity, when data is not provided."
   ":math:`c0\_min\_thresh`", ":math:`float`", "Threshold for the inlet concentration."
   ":math:`thetas`", ":math:`list[str]`", "Names of parameters to be estimated."

Degrees of Freedom
------------------
When the model is built (`IXParmest.build_model`) all inlet, design
and operating variables are fixed, leaving zero degrees of freedom.

"""

import json
import logging
import os
from copy import deepcopy
import pandas as pd
import numpy as np

import pyomo.environ as pyo
from pyomo.environ import (
    units as pyunits,
    ComponentUID,
)
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment

from idaes.core.scaling.scaling_base import ScalerBase
import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import constraint_scaling_transform

from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock

from prommis.ion_exchange.ion_exchange_multicomponent import IonExchangeMultiComp

# import prommis.ion_exchange.utils_parameter_estimation as utils_pest

log = idaeslog.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


""" This script implements an ion exchange (IX) model for parameter
estimation under the Freundlich isotherm assumption.

This example fits three IX model parameters to experimental
breakthrough data to enable the optimization of the IX column. The
three fitted parameters are: mass transfer coefficient, freundlich n
constant, and bed volumes at 50% breakthrough.

"""

path = os.path.dirname(os.path.realpath(__file__))


def main(
    regenerant: str = "single_use",
    num_traps: int = 15,
    c_trap_min: float = 1e-3,
    data_name: str = "breakthrough_literature_data",
    kt_val: float = 55,
    # n_val: float = 1.00001,
    curve_list: list | None = None,
    dir_name="parmest_results",
):
    results_parmest = []
    curve_file = f"data/{data_name}.csv"

    parmest_data = {
        "c0_ref": {},
        "loading_rate": {},
        "mass_transfer_coeff": {},
        # "freundlich_n": {},
        "bv_50": {},
    }

    for cnum in curve_list:
        c = run_parmest_workflow(
            curve_file=curve_file,
            data_name=data_name,
            curve=cnum,
            regenerant=regenerant,
            num_traps=num_traps,
            c_trap_min=c_trap_min,
            # n_val=n_val,
            kt_val=kt_val,
            dir_name=dir_name,
        )

        # Get the ix block
        ix = c.m.fs.ix  # or getattr(c.m, 'fs').ix if needed

        # Access freundlich_n
        freundlich_n = ix.freundlich_n[c.target_component].value

        # Other variables
        compound = c.component_data["name"]
        # loading_rate = c.loading_rate
        loading_rate = ix.loading_rate.value
        c0 = c.c0
        parmest_values = c.parmest_theta

        save_parmest_data(
            parmest_data,
            c,
            c0,
            loading_rate,
            freundlich_n,
            parmest_values,
        )
        print(
            f"-> Saving parameter estimation results for all components to "
            f"{dir_name}/parmest_data_updated.json"
        )

    return parmest_data


def save_parmest_data(parmest_data, c, c0, loading_rate, freundlich_n, parmest_values):
    """This function saves the calculated parameter estimation results
    for all components to a parmest_data_calculated JSON file.

    """

    comp = c.target_component

    # Initialize the compound entry
    for var in [
        "c0_ref",
        "loading_rate",
        "freundlich_n",
        "mass_transfer_coeff",
        "bv_50",
    ]:
        parmest_data.setdefault(var, {})[comp] = None

    # Populate the dictionary
    parmest_data["c0_ref"][comp] = float(c0)
    parmest_data["loading_rate"][comp] = loading_rate
    parmest_data["freundlich_n"][comp] = freundlich_n
    parmest_data["mass_transfer_coeff"][comp] = parmest_values[
        f"fs.ix.mass_transfer_coeff[{comp}]"
    ]
    parmest_data["bv_50"][comp] = parmest_values[f"fs.ix.bv_50[{comp}]"]

    # Save the parmest_data to a JSON file
    with open(f"data/parmest_data_calculated.json", "w") as json_file:
        json.dump(parmest_data, json_file, indent=4)


class IXParmest:
    def __init__(
        self,
        theta_initial_guess: dict | None = None,
        fix_vars: dict | None = None,
        **kwargs,
    ):

        self.curve_file = kwargs.get("curve_file")
        self.data_name = kwargs.get("data_name")
        self.curve = kwargs.get("curve")
        self.df_curve = self.load_curve_data(self.curve, self.data_name)
        self.df_curve.sort_values("bv", inplace=True)
        self.setup_parameters(**kwargs)

        self.m = self.build_model()

        # Initial build of model
        self.m0 = self.m.clone()
        self.R_sq = np.nan

    def load_curve_data(self, curve, data_name):
        """This method loads a CSV containing breakthrough‑curve data
        and return only the rows for the specified ``curve``.

        """

        self.curve_file = os.path.join(path, "data", f"{data_name}.csv")

        # Load the curve data.
        df = pd.read_csv(self.curve_file)

        # Filter the DataFrame to include only rows for the specified
        # target_component.
        df_filtered = df[df.curve == curve].copy()

        return df_filtered

    def setup_parameters(self, **kwargs):
        """This method sets up parameters and configuration values in
        the model.

        """

        # Parameters related to IX
        self.resin_data = kwargs.get("resin_data")
        self.resin_data_path = kwargs.get("resin_data_path")
        self.component_data = kwargs.get("component_data")
        self.regenerant = kwargs.get("regenerant")
        self.diff_calculation = kwargs.get("diff_calculation")

        # Parameters related to parameter estimation
        self.thetas = kwargs.get("thetas")
        self.initial_guess_dict = kwargs.get("initial_guess_dict")
        self.fix_initial_guess = kwargs.get("fix_initial_guess")
        self.set_bounds = kwargs.get("set_bounds")
        self.theta_initial_guess = deepcopy(self.fix_initial_guess)

        if len(self.fix_initial_guess) == 0 and len(self.theta_initial_guess) == 0:
            raise ValueError("need to provide initial guess")

        # Parameters related to plots and files
        self.dir_name = kwargs.get("dir_name")
        self.figsize = kwargs.get("figsize")
        self.all_figs = dict()

        self.scale_from_value = kwargs.get("scale_from_value")

        # Get conditions from breakthrough curve data file. This
        # function also includes the ix_config values like diffusivity
        # and ions_props values.
        self.get_curve_conditions(
            num_traps=kwargs.get("num_traps"),
            c_trap_min=kwargs.get("c_trap_min"),
        )

        # Print curve values
        print("\n================================================")
        print(f"Curve number: {self.curve}")
        print(f"Compound: {self.target_component}")
        print(f"Resin: {self.resin.upper()}")
        print(f"EBCT: {self.ebct_min} min")
        print(f"Reference: {self.ref.replace('_', ' ').title()}")
        print("================================================\n")

    def get_curve_conditions(self, num_traps, c_trap_min):
        """This method gets the conditions for the breakthrough curve
        to use in model and other metadata.

        """

        mw_water = 0.018 * pyunits.kg / pyunits.mol
        rho = 1000 * pyunits.kg / pyunits.m**3

        df = self.df_curve.copy()

        self.ref = df.ref.iloc[0]
        self.c0 = df.c0.iloc[0]

        self.flow_in = df.flow_in.iloc[0]  # in m3/s
        self.loading_rate = df.vel_bed.iloc[0]  # in m/s
        self.bed_depth = df.bed_depth.iloc[0] * pyunits.m
        self.bed_diam = df.bed_diam.iloc[0] * pyunits.m
        self.bed_volume_total = df.bed_vol.iloc[0] * pyunits.m**3
        self.resin = df.resin.iloc[0]
        self.resin_type = self.resin_data[self.resin]["type"]
        self.resin_func_group = self.resin_data[self.resin]["functional_group"]
        self.ebct_min = float(df.ebct.iloc[0])
        self.conc_units_str = df.conc_units.iloc[0]
        numerator = getattr(pyunits, self.conc_units_str.split("/")[0])
        denominator = getattr(pyunits, self.conc_units_str.split("/")[1])
        self.conc_units = numerator / denominator
        self.target_component = df.compound.iloc[0]
        self.num_traps = num_traps
        self.c_trap_min = c_trap_min
        self.charge = self.component_data["charge"]
        self.mw = self.component_data["mw"] * pyunits.kg / pyunits.mol
        self.resin_density = self.resin_data[self.resin]["density_kgm3"]["value"]
        self.resin_diam = self.resin_data[self.resin]["diameter"]["value"]
        self.resin_porosity = self.resin_data[self.resin]["porosity"]["value"]
        self.molar_vol = self.component_data["molar_vol"]
        self.diffusivity = self.component_data["diffusivity"]

        # Get original breakthrough data.
        self.get_bvs_cbs_breakthrough()

        # Calculate initial values.
        self.c0_mol_flow = pyunits.convert(
            (self.c0 * self.conc_units)
            / self.mw
            * (self.flow_in * pyunits.m**3 / pyunits.s),
            to_units=pyunits.mol / pyunits.s,
        )
        self.c0_mol_flow_sf = 1 / pyo.value(self.c0_mol_flow)
        self.water_mol_flow = pyunits.convert(
            ((self.flow_in * pyunits.m**3 / pyunits.s) * rho) / mw_water,
            to_units=pyunits.mol / pyunits.s,
        )
        self.water_mol_flow_sf = 1 / self.water_mol_flow()

        self.cb = self.keep_cbs[-1] * self.conc_units
        self.conc_comp_norm_breakthrough = self.keep_cnorms[-1]
        self.bv = self.keep_bvs[-1]

        self.fix_initial_guess["bv_50"] = np.mean(self.keep_bvs)
        self.theta_initial_guess["bv_50"] = np.mean(self.keep_bvs)

        # Declare all the configuration arguments in the ion exchange
        # unit model.
        if self.diff_calculation != "HaydukLaudie":
            self.ion_props = {
                "solute_list": [self.target_component],
                "diffusivity_data": {("Liq", self.target_component): self.diffusivity},
                "mw_data": {"H2O": 0.018015, self.target_component: self.mw},
                "charge": {self.target_component: self.charge},
            }
        else:
            self.ion_props = {
                "solute_list": [self.target_component],
                "mw_data": {"H2O": mw_water, self.target_component: self.mw},
                "diffus_calculation": self.diff_calculation,
                "charge": {self.target_component: self.charge},
                "molar_volume_data": {("Liq", self.target_component): self.molar_vol},
            }

        self.ix_config = {
            "regenerant": self.regenerant,
            "target_component": self.target_component,
            "reactive_ions": [self.target_component],
            "number_of_trapezoids": self.num_traps,
            "minimum_concentration_trapezoids": self.c_trap_min,
            "resin_data_path": self.resin_data_path,
            "resin": self.resin,
        }

    def build_model(self):
        """This method builds the IX model."""

        m = pyo.ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(**self.ion_props)

        self.ix_config["property_package"] = m.fs.properties

        m.fs.ix = ix = IonExchangeMultiComp(**self.ix_config)

        self.add_bounds(m)
        self.add_operating_conditions(m)

        self.ix = self._get_ix_blk(m)
        self.scale_it(m)

        # For parameter estimation, always deactivate the effluent
        # calculation constraints.
        self.deactivate_trapezoids(m)

        assert degrees_of_freedom(m) == 0
        self.add_initialization(m)

        return m

    def add_bounds(self, m):
        """This method sets the bounds for the IX model variables."""

        ix = m.fs.ix
        pf_in = m.fs.ix.process_flow.properties_in[0.0]
        pf_out = m.fs.ix.process_flow.properties_out[0.0]

        pf_in.flow_vol_phase["Liq"].setlb(1e-20)
        pf_in.visc_k_phase.setlb(1e-20)
        pf_in.diffus_phase_comp["Liq", self.target_component].setlb(1e-22)
        pf_in.flow_mass_phase_comp["Liq", self.target_component].setlb(1e-22)
        pf_out.flow_mass_phase_comp["Liq", self.target_component].setlb(1e-22)
        pf_in.flow_mol_phase_comp["Liq", self.target_component].setlb(1e-22)

        for k, v in self.set_bounds.items():
            ixv = getattr(ix, k)

            # Check if ixv is an Expression and skip setting bounds.
            if isinstance(ixv, pyo.Expression):
                continue

            # Check if ixv is a Var().
            if isinstance(ixv, pyo.Var):
                # Check if the bounds are indexed.
                if isinstance(v, dict):  # If v is a dictionary, it contains indexed
                    # bounds.
                    for index, bounds in v.items():
                        # Set lower bound for the indexed variable.
                        ixv[index].setlb(bounds[0])
                        # Set upper bound for the indexed variable.
                        ixv[index].setub(bounds[1])
                else:
                    # If v is not a dictionary, set bounds directly.
                    ixv.setlb(v[0])
                    ixv.setub(v[1])
            else:
                print(
                    f"Skipping {k} as it is neither a Pyomo Var() nor an Expression()."
                )

    def add_operating_conditions(self, m):
        """Add data to the model."""

        ix = m.fs.ix
        pf_in = ix.process_flow.properties_in[0]

        # Fix inlet conditions in IX.
        pf_in.flow_mol_phase_comp["Liq", "H2O"].fix(self.water_mol_flow)
        pf_in.flow_mol_phase_comp["Liq", self.target_component].fix(self.c0_mol_flow)
        pf_in.pressure.fix(101325)
        pf_in.temperature.fix(298.15)

        # Fix design and operational parameters in IX.
        ix.resin_diam.fix(self.resin_diam)
        ix.resin_density.fix(self.resin_density)
        ix.bed_depth.fix(self.bed_depth)
        ix.bed_diameter.fix(self.bed_diam)
        ix.bed_porosity.fix(0.8)
        ix.number_columns.fix(1)
        ix.number_columns_redundant.fix(1)
        ix.loading_rate.set_value(self.loading_rate)  # in m/s
        ebct_s = pyo.value(
            pyo.units.convert(self.ebct_min * pyo.units.minute, to_units=pyo.units.s)
        )
        ix.ebct.set_value(ebct_s)  # in s
        ix.conc_comp_norm_breakthrough[self.target_component].fix(0.80)

        ix.freundlich_n[self.target_component].fix(1.0001)
        ix.bv[self.target_component].set_value(250)

        # Fix initial guess for theta parameters.
        if isinstance(self.fix_initial_guess, dict):
            for k, v in self.fix_initial_guess.items():
                ixv = getattr(ix, k)
                ixv.fix(v)

    def scale_it(self, m=None):
        """This function scales all variables in the model"""

        ix = self._get_ix_blk(m=m)

        comp = ix.config.target_component

        sb = ScalerBase()

        for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
            if "temperature" in var.name:
                sb.set_variable_scaling_factor(var, 1e-1, overwrite=True)
            if "pressure" in var.name:
                sb.set_variable_scaling_factor(var, 1e-5)
            if "flow_vol" in var.name:
                sb.set_variable_scaling_factor(var, 1e8)
            if "flow_mol_phase_comp" in var.name:
                if "H2O" in var.name:
                    sb.set_variable_scaling_factor(var, self.water_mol_flow_sf)
                else:
                    sb.set_variable_scaling_factor(var, self.c0_mol_flow_sf)
            if "conc_comp_norm_breakthrough" in var.name:
                if "H2O" in var.name:
                    sb.set_variable_scaling_factor(var, 1e-3)
                else:
                    sb.set_variable_scaling_factor(var, 1e6)
            if "breakthrough_time" in var.name:
                sb.set_variable_scaling_factor(var, 1e-6)

        if isinstance(self.scale_from_value, list):
            for v in self.scale_from_value:
                ixv = getattr(ix, v)
                if ixv.is_indexed():
                    idx = [*ixv.index_set()]
                    for i in idx:
                        if ixv[i]() == 0:
                            continue
                        sb.set_variable_scaling_factor(ixv[i], 1 / ixv[i]())
                else:
                    if ixv() == 0:
                        continue
                    sb.set_variable_scaling_factor(ixv, 1 / ixv())

        constraint_scaling_transform(ix.eq_clark[comp], 1e-1)

    def deactivate_trapezoids(self, m):
        """This method deactivates all equations related to the
        trapezoid rule since we are fitting data to C/C0 and do not
        care about the steady-state effluent concentration.

        """

        ix = m.fs.ix
        pf = ix.process_flow

        ix.eq_breakthrough_time_trapezoids.deactivate()
        ix.eq_conc_comp_norm_trapezoids.deactivate()
        ix.eq_trapezoids.deactivate()
        ix.eq_conc_comp_norm_breakthrough_trapezoids.deactivate()
        for _, v in ix.breakthrough_time_trapezoids.items():
            v.fix()
        for _, v in ix.conc_comp_norm_trapezoids.items():
            v.fix()
        for _, v in ix.trapezoids.items():
            v.fix()
        ix.conc_comp_norm_breakthrough_trapezoids.fix()
        ix.eq_mass_transfer_reactive_ions.deactivate()
        pf.mass_transfer_term[0, "Liq", self.target_component].fix()

    def _calc_from_constr(self, m):

        ix = m.fs.ix

        if not hasattr(self, "calc_from_constr_fail"):
            self.calc_from_constr_fail = dict()

    def _get_ix_blk(self, m=None):

        for blk in m.fs.component_objects(pyo.Block):
            if isinstance(blk, IonExchangeMultiComp):
                return blk

    def add_initialization(self, m):

        m.fs.ix.initialize(outlvl=idaeslog.INFO)

    # def run_all_things(self, plot_things, save_things, overwrite=True):
    def run_all_things(self, overwrite=True):
        """Run all methods for parmest."""

        # Add solve options.
        solver_options = {
            "print_level": 5,
            "halt_on_ampl_error": "yes",
            "nlp_scaling_method": "gradient-based",
            "max_iter": 1000,
            "tol": 1e-6,
            "bound_push": 1e-8,
            "constr_viol_tol": 1e-6,
            "acceptable_constr_viol_tol": 1e-6,
        }

        self.run_parmest(solver_options)

        # utils_pest.plot_estimated_curve(self, solver_options)
        # os.makedirs(self.dir_name, exist_ok=True)
        # utils_pest.save_figs(self)

    def get_bvs_cbs_breakthrough(self):
        """This method saves extracted breakthrough data."""

        df = self.df_curve.copy().dropna()
        last_cb = -0.01
        self.keep_cbs = []
        self.keep_bvs = []
        self.excl_cbs = []
        self.excl_bvs = []
        self.all_bvs = df.bv.to_list()
        self.all_cbs = df.cb.to_list()
        self.all_cnorms = df.c_norm.to_list()

        print("INFO: Using all data")
        tmp = df[df.c_norm != 0].copy()
        self.keep_bvs = tmp.bv.to_list()
        self.keep_cbs = tmp.cb.to_list()
        assert 0 not in self.keep_cbs

        self.keep_cnorms = [cb / self.c0 for cb in self.keep_cbs]
        self.excl_cnorms = [cb / self.c0 for cb in self.excl_cbs]

    def build_parmest(self):
        """This method constructs the parmest model for the selected
        breakthrough data.

        """

        data = self.df_curve[self.df_curve.cb.isin(self.keep_cbs)]
        cb = pyunits.convert(
            data.cb.to_list()[0] * self.conc_units,
            to_units=pyunits.kg / pyunits.m**3,
        )()
        cb_p = data.cb.to_list()[0] * self.conc_units
        c0_p = self.c0 * self.conc_units
        cx = pyo.value(cb_p / c0_p)
        bv = data.bv.to_list()[0]

        m_parmest = self.m.clone()
        m_parmest.fs.ix.conc_comp_norm_breakthrough.fix(cx)

        m_parmest.compound = data.compound.tolist()[0]

        self.scale_it(m=m_parmest)

        return m_parmest

    def build_parmest_experiment_list(self):
        """This method assembles a list of `Experiment` objects for
        parmest.

        """

        self.experiment_list = list()
        exp_dict = {
            "exp_num": [x for x in range(len(self.keep_bvs))],
            "cb": self.keep_cbs,
            "bv": self.keep_bvs,
            "conc_comp_norm_breakthrough": self.keep_cnorms,
        }
        self.df_exp = pd.DataFrame.from_dict(exp_dict).set_index("exp_num")
        self.df_exp["c0"] = self.c0
        for i in self.df_exp.index:
            exper = IXExperiment(self, self.df_exp, i)
            self.experiment_list.append(exper)
            # print('in exp list:', exper.data_i)

    def run_parmest(self, solver_options):
        """Run parmest"""

        print(f"\nINFO: Running parmest...")

        if not hasattr(self, "experiment_list"):
            self.build_parmest_experiment_list()

        # Create an instance of the Parmest Estimator and solve.
        self.pestimator = parmest.Estimator(
            self.experiment_list,
            obj_function="SSE",
            diagnostic_mode=False,
            tee=True,
            solver_options=solver_options,
        )

        n = len(self.experiment_list)
        self.obj, self.parmest_theta = self.pestimator.theta_est()

        self.cov = self.pestimator.cov_est()

        # Print the values of the fitted parameters.
        print()
        print("==============================")
        print(f"Parmest Results for {self.target_component} in curve {self.curve}")
        print()
        print(f"Objective: {self.obj}")
        print(f"{self.parmest_theta}")
        print()
        print("==============================")

        # # # Create a new directory to save the results and also access
        # # the estimated parameters from regression results and save
        # # them in a new dictionary theta_dict. The estimated
        # # parameters in this dictionary are used to calculate the
        # # estimated curve in plot_estimated_curve.
        # utils_pest.create_results_directory(self)
        # utils_pest.extract_and_save_theta(self)


class IXExperiment(Experiment):
    def __init__(self, ix_parmest_obj, data, experiment_number, meas_error=None):
        self.data = data
        self.experiment_number = experiment_number
        self.data_i = data.loc[experiment_number]
        self.ix_parmest_obj = ix_parmest_obj
        self.model = None
        self.thetas = self.ix_parmest_obj.thetas
        self.meas_error = meas_error

    def create_model(self):
        self.model = self.ix_parmest_obj.build_parmest()

    def finalize_model(self):

        m = self.model
        cb = float(self.data_i.cb)
        c0 = float(self.data_i.c0)

        # Add target_component index
        self.conc_comp_norm_breakthrough = cb / c0
        m.fs.ix.conc_comp_norm_breakthrough.fix(self.conc_comp_norm_breakthrough)

        for k in self.ix_parmest_obj.initial_guess_dict.keys():
            ixv = getattr(self.model.fs.ix, k)
            ixv.unfix()

        return m

    def label_model(self):

        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        # Get the target component
        target_component = m.fs.ix.config.target_component

        m.experiment_outputs.update(
            [
                (
                    m.fs.ix.conc_comp_norm_breakthrough[target_component],
                    self.data_i["conc_comp_norm_breakthrough"],
                ),
                (m.fs.ix.bv[target_component], self.data_i["bv"]),
            ]
        )

        m.measurement_error.update(
            [
                (
                    m.fs.ix.conc_comp_norm_breakthrough[target_component],
                    self.meas_error,
                ),
                (m.fs.ix.bv[target_component], self.meas_error),
            ]
        )

        m.unknown_parameters.update(
            [
                # (
                #     m.fs.ix.freundlich_n[target_component],
                #     ComponentUID(m.fs.ix.freundlich_n[target_component])
                # ),
                (
                    m.fs.ix.mass_transfer_coeff[target_component],
                    ComponentUID(m.fs.ix.mass_transfer_coeff[target_component]),
                ),
                (
                    m.fs.ix.bv_50[target_component],
                    ComponentUID(m.fs.ix.bv_50[target_component]),
                ),
            ]
        )

        return m

    def get_labeled_model(self):

        m = self.create_model()
        m = self.finalize_model()
        m = self.label_model()

        return m


def run_parmest_workflow(
    curve_file,
    data_name,
    curve,
    regenerant,
    num_traps,
    c_trap_min,
    # n_val,
    kt_val,
    dir_name,
):

    df_ree = pd.read_csv(curve_file)
    set_bounds = {
        "mass_transfer_coeff": [1e-3, 1e2],
        # "freundlich_n": [1, 1e3],
        "bv_50": [0, 1e3],
        "bv": [1e-3, 1e3],
        "service_flow_rate": [0, 1e6],
        "ebct": [0, 1e6],
        "bed_diameter": [0.01, 100],
        "bed_depth": [0.001, 100],
        "loading_rate": [0, 1e6],
    }
    scale_from_value = [
        "bed_volume_total",
        "column_height",
        "service_flow_rate",
        "ebct",
        "bv_50",
        "resin_diam",
        "resin_density",
        "bed_porosity",
        "bed_diameter",
        "bed_volume",
        "bed_volume_total",
        "bed_depth",
        "number_columns",
        "number_columns_redundant",
        "conc_comp_norm_breakthrough",
        # "freundlich_n",
        "mass_transfer_coeff",
        "loading_rate",
        "bv",
        "N_Re",
        "N_Pe_particle",
        "N_Pe_bed",
        "breakthrough_time",
    ]

    # IX parameters to estimate.
    thetas = [
        # "freundlich_n",
        "mass_transfer_coeff",
        "bv_50",
    ]

    input_data = df_ree[df_ree.curve == curve].copy()

    resin_file = os.path.join(path, "data", "resin_data.json")
    with open(resin_file, "r") as file:
        resin_data = json.load(file)

    comp_prop_file = os.path.join(path, "data", "properties_data.json")
    with open(comp_prop_file) as data_file:
        species_props = json.load(data_file)

    species = input_data.compound.iloc[0]
    component_data = dict(
        name=species,
        mw=float(species_props["mw"][species]),
        charge=species_props["charge"][species],
        diffusivity=5e-10,
        molar_vol=float(species_props["molar_vol"][species]),
    )

    # By default, bv_50 has no initial guess. Default initial guess
    # for bv_50 is avg of BV values from curve.
    initial_guess_dict = {
        # "freundlich_n": n_val,
        "mass_transfer_coeff": kt_val,
    }
    fix_initial_guess = {
        # "freundlich_n": n_val,
        "mass_transfer_coeff": kt_val,
    }

    c = IXParmest(
        curve_file=curve_file,
        data_name=data_name,
        curve=curve,
        num_traps=num_traps,  # number of trapezoids
        c_trap_min=c_trap_min,  # initial concentration of target component
        resin_data=resin_data,
        resin_data_path=resin_file,
        initial_guess_dict=initial_guess_dict,
        set_bounds=set_bounds,
        component_data=component_data,
        regenerant=regenerant,
        fix_initial_guess=fix_initial_guess,
        scale_from_value=scale_from_value,
        diff_calculation="HaydukLaudie",
        c0_min_thresh=0,
        thetas=thetas,
        figsize=(10, 8),
        dir_name=dir_name,
    )

    c.run_all_things()

    return c


if __name__ == "__main__":

    curve_list = [1, 2, 3, 4, 5, 6]
    main(curve_list=curve_list)
