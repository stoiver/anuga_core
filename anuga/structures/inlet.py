
import anuga.geometry.polygon
from anuga.geometry.polygon import inside_polygon, is_inside_polygon, line_intersect
from anuga.config import velocity_protection, g
from anuga import Region

import math


import numpy as num

def shift_depths_to_average(depths, areas, old_average_depth, new_average_depth):
    """Shift an inlet's water surface to a new average depth, keeping its shape.

    Returns the new per-cell depths. The surface is moved by a single uniform
    offset (``new_average_depth - old_average_depth``) rather than being
    flattened to a uniform depth, so a level surface stays level: on a sloping
    bed, writing a uniform DEPTH tilts the water surface to follow the bed and
    disturbs a lake at rest by half the bed elevation range across the inlet
    (issue #229).

    Wet/dry clamp: cells the offset would drive below their bed are clamped to
    zero depth, and the water that could not be taken from them is taken from
    the cells that are still wet instead, repeatedly until every cell is
    non-negative. So the total volume change is exactly
    ``(new_average_depth - old_average_depth) * total_area`` whenever the inlet
    holds enough water to supply it, and the whole inlet goes dry only when it
    does not.

    Parameters
    ----------
    depths, areas : array
        Current per-cell depth and area.
    old_average_depth : float
        The average depth the caller's `new_average_depth` was derived from. In
        parallel this is the GLOBAL average over the whole inlet, while `depths`
        and `areas` are this rank's share, so it is passed in rather than
        recomputed here.
    new_average_depth : float
        Target average depth.
    """

    depths = num.asarray(depths, dtype=float)
    areas = num.asarray(areas, dtype=float)

    shift = new_average_depth - old_average_depth
    new_depths = depths + shift
    if new_depths.min() >= 0.0:
        return new_depths

    # Draining more than some cells hold. Take the shortfall from the cells that
    # are still wet; that may dry out more of them, so repeat. Each pass dries at
    # least one cell, so this terminates in at most len(depths) passes.
    for _ in range(len(depths)):
        dry = new_depths < 0.0
        # Volume we failed to remove by clamping those cells at their bed.
        deficit = -float(num.dot(new_depths[dry], areas[dry]))
        new_depths[dry] = 0.0
        wet = new_depths > 0.0
        wet_area = float(areas[wet].sum())
        if deficit <= 0.0 or wet_area <= 0.0:
            break
        new_depths[wet] -= deficit / wet_area
        if new_depths.min() >= 0.0:
            break

    return num.maximum(new_depths, 0.0)


class Inlet:
    """Contains information associated with each inlet
    """

    def __init__(self, domain, poly, verbose=False):

        self.domain = domain
        self.domain_bounding_polygon = self.domain.get_boundary_polygon()
        self.verbose = verbose


        # poly can be either a line, polygon or a region
        if isinstance(poly,Region):
            self.region = poly
        else:
            self.region = Region(domain,poly=poly,expand_polygon=True)

        self.triangle_indices = self.region.indices

        self.compute_area()


    def compute_area(self):

        # Compute inlet area as the sum of areas of triangles identified
        # by line. Must be called after compute_inlet_triangle_indices().
        if len(self.triangle_indices) == 0:
            region = 'Inlet line=%s' % (self.inlet_line)
            msg = 'No triangles have been identified in region '
            raise Exception(msg)

#        self.area = 0.0
#        for j in self.triangle_indices:
#            self.area += self.domain.areas[j]

        self.area = num.sum(self.domain.areas[self.triangle_indices])

        msg = 'Inlet exchange area has area = %f' % self.area
        assert self.area > 0.0



    def get_poly(self):

        return self.poly

    def get_area(self):

        return self.area


    def get_areas(self):

        # Must be called after compute_inlet_triangle_indices().
        return self.domain.areas.take(self.triangle_indices)


    def get_stages(self):

        return self.domain.quantities['stage'].centroid_values.take(self.triangle_indices)
        # self.domain.quantities['stage'].centroid_values would be called gpu_stage_centroid_values in the gpu_interface

    def get_average_stage(self):

        return num.sum(self.get_stages()*self.get_areas())/self.area

    def get_elevations(self):

        return self.domain.quantities['elevation'].centroid_values.take(self.triangle_indices)

    def get_average_elevation(self):

        return num.sum(self.get_elevations()*self.get_areas())/self.area


    def get_xmoms(self):

        return self.domain.quantities['xmomentum'].centroid_values.take(self.triangle_indices)


    def get_average_xmom(self):

        return num.sum(self.get_xmoms()*self.get_areas())/self.area


    def get_ymoms(self):

        return self.domain.quantities['ymomentum'].centroid_values.take(self.triangle_indices)


    def get_average_ymom(self):

        return num.sum(self.get_ymoms()*self.get_areas())/self.area


    def get_depths(self):

        return self.get_stages() - self.get_elevations()


    def get_total_water_volume(self):

       return num.sum(self.get_depths()*self.get_areas())


    def get_average_depth(self):

        return self.get_total_water_volume()/self.area


    def get_velocities(self):

            depths = self.get_depths()
            u = self.get_xmoms()*depths/(depths*depths + velocity_protection)
            v = self.get_ymoms()*depths/(depths*depths + velocity_protection)

            return u, v


    def get_xvelocities(self):

            depths = self.get_depths()
            return self.get_xmoms()*depths/(depths*depths + velocity_protection)

    def get_yvelocities(self):

            depths = self.get_depths()
            return self.get_ymoms()*depths/(depths*depths + velocity_protection)


    def get_average_speed(self):

            u, v = self.get_velocities()

            average_u = num.sum(u*self.get_areas())/self.area
            average_v = num.sum(v*self.get_areas())/self.area

            return math.sqrt(average_u**2 + average_v**2)


    def get_average_velocity_head(self):

        return 0.5*self.get_average_speed()**2/g


    def get_average_total_energy(self):

        return self.get_average_velocity_head() + self.get_average_stage()


    def get_average_specific_energy(self):

        return self.get_average_velocity_head() + self.get_average_depth()



    def set_depths(self,depth):

        self.domain.quantities['stage'].centroid_values.put(self.triangle_indices, self.get_elevations() + depth)


    def set_stages(self,stage):

        self.domain.quantities['stage'].centroid_values.put(self.triangle_indices, stage)

    def set_average_depth(self, new_average_depth, old_average_depth=None):
        """Move the water surface to give `new_average_depth`, keeping its shape.

        The structure operators' way of writing back a transfer. Unlike
        set_depths(), this shifts the surface instead of flattening it, so it is
        well balanced on a sloping bed, and it clamps cells at their bed rather
        than driving them to negative depth. See shift_depths_to_average().
        """

        if old_average_depth is None:
            old_average_depth = self.get_average_depth()

        new_depths = shift_depths_to_average(
            self.get_depths(), self.get_areas(),
            old_average_depth, new_average_depth)
        self.set_stages(self.get_elevations() + new_depths)


    def set_xmoms(self,xmom):

        self.domain.quantities['xmomentum'].centroid_values.put(self.triangle_indices, xmom)


    def set_ymoms(self,ymom):

        self.domain.quantities['ymomentum'].centroid_values.put(self.triangle_indices, ymom)


    def set_elevations(self,elevation):

        self.domain.quantities['elevation'].centroid_values.put(self.triangle_indices, elevation)

    def set_stages_evenly(self,volume):
        """ Distribute volume of water over
        inlet exchange region so that stage is level
        """

        assert volume >= 0.0

        areas = self.get_areas()
        stages = self.get_stages()
        depths = self.get_depths()

        elevations = self.get_elevations()

        #print('elevation')
        #print(elevations)

        stages_order = stages.argsort()

        # accumulate areas of cells ordered by stage
        summed_areas = num.cumsum(areas[stages_order])

        # accumulate the volume need to fill cells
        summed_volume = num.zeros_like(areas)
        summed_volume[1:] = num.cumsum(summed_areas[:-1]*num.diff(stages[stages_order]))

        index = num.nonzero(summed_volume<=volume)[0][-1]

        # calculate stage needed to fill chosen cells with given volume of water
        depth = (volume - summed_volume[index])/summed_areas[index]
        stages[stages_order[0:index+1]] = stages[stages_order[index]]+depth

        #print('stages')
        #print(stages)
        self.set_stages(stages)




    def set_depths_evenly(self,volume):
        """ Distribute volume over all exchange
        cells with equal depth of water
        """

        new_depth = self.get_average_depth() + (volume/self.get_area())
        self.set_depths(new_depth)

