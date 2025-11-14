import geopandas as gpd
import pandas as pd
import re
import os
from shapely.validation import make_valid
from shapely.geometry import Point, box
import warnings
warnings.filterwarnings('ignore')


class SeattleParkingCleaner:
    """Cleaner for Seattle parking datasets"""
    
    def __init__(self, raw_dir='raw_data', clean_dir='assets'):
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)
    
    def clean_blockface_comprehensive(self, filename='parking_categories.geojson'):
        """Clean the comprehensive blockface dataset with all parking info"""   
        try:
            filepath = os.path.join(self.raw_dir, filename)
            df = gpd.read_file(filepath)
            print(f"  ✓ Loaded {len(df)} features")

            df = df[df.geometry.notna()]
            df['geometry'] = df.geometry.apply(
                lambda g: make_valid(g) if not g.is_valid else g
            )
            print(f"  ✓ Fixed geometries")

            original_count = len(df)
            df = df[
                (df['PARKING_CATEGORY'].notna()) | 
                (df['PARKING_SPACES'] > 0)
            ]
            print(f"  ✓ Filtered to {len(df)} segments with parking ({len(df)/original_count*100:.1f}%)")

            df['category_std'] = df['PARKING_CATEGORY'].fillna('UNKNOWN').str.strip().str.upper()

            category_mapping = {
                'PAID PARKING': 'PAID',
                'PAID': 'PAID',
                'UNRESTRICTED': 'UNRESTRICTED',
                'RESTRICTED': 'RESTRICTED',
                'CARPOOL': 'CARPOOL',
                'RPZ': 'PERMIT',
                'TIME LIMIT': 'TIME_LIMITED',
                'NO PARKING': 'NO_PARKING'
            }
            
            df['category_clean'] = df['category_std'].map(category_mapping).fillna(df['category_std'])
            print(f"  ✓ Standardized categories: {df['category_clean'].unique()}")

            df['time_limit_minutes'] = df['PARKING_TIME_LIMIT'].apply(self._parse_time_limit)
            print(f"  ✓ Parsed time limits")
   
            df['total_spaces'] = pd.to_numeric(df['TOTAL_SPACES'], errors='coerce').fillna(0).astype(int)
            df['paid_spaces'] = pd.to_numeric(df['PAID_SPACES'], errors='coerce').fillna(0).astype(int)
            df['unrestricted_spaces'] = pd.to_numeric(df['UNRESTRICTED'], errors='coerce').fillna(0).astype(int)
            print(f"  ✓ Cleaned space counts")

            df['weekday_rate'] = df['WKD_RATE1'].apply(self._clean_currency)
            df['weekday_start'] = df['START_TIME_WKD'].apply(self._standardize_time)
            df['weekday_end'] = df['END_TIME_WKD'].apply(self._standardize_time)
            print(f"  ✓ Cleaned weekday pricing")

            df['price_category'] = df['weekday_rate'].apply(self._categorize_price)

            df['has_paid_parking'] = df['paid_spaces'] > 0
            df['is_permit_zone'] = df['RPZ_ZONE'].notna()
            df['is_peak_hour_restricted'] = df['PEAK_HOUR'].notna()
            print(f"  ✓ Created boolean flags")
            
            df['parking_type'] = 'street'

            df['block_id'] = df['BLOCK_ID'].fillna('')
            df['side'] = df['SIDE'].fillna('').str.upper()
            
            print(f"  → Simplifying geometries...")
            df['geometry'] = df.geometry.simplify(tolerance=0.00005, preserve_topology=True)
            print(f"  ✓ Simplified geometries")

            output_cols = [
                'geometry',
                'OBJECTID',
                'parking_type',
                'category_clean',
                'time_limit_minutes',
                'total_spaces',
                'paid_spaces',
                'unrestricted_spaces',
                'weekday_rate',
                'weekday_start',
                'weekday_end',
                'price_category',
                'has_paid_parking',
                'is_permit_zone',
                'is_peak_hour_restricted',
                'block_id',
                'side'
            ]
            
            df_clean = df[output_cols]
            df_clean = df_clean.drop_duplicates(subset=['geometry'])
            output_path = os.path.join(self.clean_dir, 'street_parking_detailed.geojson')
            df_clean.to_crs('EPSG:4326').to_file(output_path, driver='GeoJSON')
            print(f"  ✓ Exported {len(df_clean)} features to street_parking_detailed.geojson")
            self._print_file_size(output_path)
            print(f"  → Creating overview dataset...")
            overview = self._create_parking_overview(df_clean)
            if overview is not None:
                overview_path = os.path.join(self.clean_dir, 'street_parking_overview.geojson')
                overview.to_file(overview_path, driver='GeoJSON')
                print(f"  ✓ Created overview with {len(overview)} grid cells")
                self._print_file_size(overview_path)
            
            return df_clean
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def clean_parking_tiers(self, filename='parking_tiers.geojson'):
        """Clean parking tiers if available as separate file"""
        
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"  ⊘ File not found: {filename} (skipping)")
            return None
        
        try:
            df = gpd.read_file(filepath)
            print(f"  ✓ Loaded {len(df)} features")

            df = df[df.geometry.notna()]
            df['geometry'] = df.geometry.apply(
                lambda g: make_valid(g) if not g.is_valid else g
            )

            rate_columns = [col for col in df.columns if 'RATE' in col.upper() or 'PRICE' in col.upper()]
            
            for col in rate_columns:
                new_col = col.lower().replace(' ', '_')
                df[new_col] = df[col].apply(self._clean_currency)
            
            time_columns = [col for col in df.columns if 'START' in col.upper() or 'END' in col.upper()]
            
            for col in time_columns:
                new_col = col.lower().replace(' ', '_') + '_std'
                df[new_col] = df[col].apply(self._standardize_time)

            numeric_rates = [col for col in df.columns if 'rate' in col.lower() and df[col].dtype in ['float64', 'int64']]
            if numeric_rates:
                df['avg_rate'] = df[numeric_rates].mean(axis=1)
                df['price_category'] = df['avg_rate'].apply(self._categorize_price)
            df['geometry'] = df.geometry.simplify(tolerance=0.0001, preserve_topology=True)
            output_path = os.path.join(self.clean_dir, 'parking_tiers_clean.geojson')
            df.to_crs('EPSG:4326').to_file(output_path, driver='GeoJSON')
            
            print(f"  ✓ Exported {len(df)} features")
            self._print_file_size(output_path)
            
            return df
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def clean_garages(self, filename='garages.geojson'):
        """Clean public garages dataset"""
        
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"  ⊘ File not found: {filename} (skipping)")
            return None
        
        try:
            df = gpd.read_file(filepath)
            print(f"  ✓ Loaded {len(df)} features")
            
            df = df[df.geometry.notna()]
            df = df[df.geometry.type == 'Point']

            name_col = None
            for col in ['NAME', 'GARAGE_NAME', 'LOCATION_NAME', 'FACILITY_NAME']:
                if col in df.columns:
                    name_col = col
                    break
            
            if name_col:
                df['name'] = df[name_col].str.strip()
            capacity_col = None
            for col in ['CAPACITY', 'SPACES', 'TOTAL_SPACES', 'NUM_SPACES']:
                if col in df.columns:
                    capacity_col = col
                    break
            
            if capacity_col:
                df['capacity'] = df[capacity_col].apply(self._clean_capacity)
            hours_col = None
            for col in ['HOURS', 'OPERATING_HOURS', 'OPEN_HOURS']:
                if col in df.columns:
                    hours_col = col
                    break
            
            if hours_col:
                df['hours'] = df[hours_col].fillna('24/7').astype(str).str.strip()
                df['is_24_7'] = df['hours'].str.contains('24', case=False, na=False)
            address_col = None
            for col in ['ADDRESS', 'LOCATION', 'STREET_ADDRESS']:
                if col in df.columns:
                    address_col = col
                    break
            
            if address_col:
                df['address'] = df[address_col].str.strip()

            df['parking_type'] = 'garage'

            output_cols = ['geometry', 'parking_type']
            for col in ['name', 'capacity', 'hours', 'is_24_7', 'address', 'OBJECTID']:
                if col in df.columns:
                    output_cols.append(col)
            
            df_clean = df[output_cols]
            output_path = os.path.join(self.clean_dir, 'garages_clean.geojson')
            df_clean.to_crs('EPSG:4326').to_file(output_path, driver='GeoJSON')
            
            print(f"  ✓ Exported {len(df_clean)} features")
            self._print_file_size(output_path)
            
            return df_clean
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def create_combined_dataset(self):
        """Create a combined points dataset for quick overview"""
        
        try:
            street_path = os.path.join(self.clean_dir, 'street_parking_detailed.geojson')
            garage_path = os.path.join(self.clean_dir, 'garages_clean.geojson')
            
            if not os.path.exists(street_path):
                print("  ⊘ Street parking not found, skipping combined dataset")
                return None

            street = gpd.read_file(street_path)
            street_points = street.copy()
            street_points['geometry'] = street_points.geometry.centroid

            street_simple = street_points[['geometry', 'parking_type', 'category_clean', 
                                          'total_spaces', 'price_category']].copy()
            street_simple = street_simple.rename(columns={'category_clean': 'category'})
            if os.path.exists(garage_path):
                garages = gpd.read_file(garage_path)
                garage_simple = garages[['geometry', 'parking_type']].copy()
                garage_simple['category'] = 'GARAGE'
                if 'capacity' in garages.columns:
                    garage_simple['total_spaces'] = garages['capacity']
                else:
                    garage_simple['total_spaces'] = 0
                garage_simple['price_category'] = 'UNKNOWN'
                combined = pd.concat([street_simple, garage_simple], ignore_index=True)
            else:
                combined = street_simple
                print("  ⊘ Garages not found, using only street parking")

            combined = gpd.GeoDataFrame(combined, crs='EPSG:4326')
            combined_sample = combined.iloc[::3].copy()
            output_path = os.path.join(self.clean_dir, 'parking_all_points.geojson')
            combined_sample.to_file(output_path, driver='GeoJSON')
            
            print(f"  ✓ Created combined dataset with {len(combined_sample)} points")
            self._print_file_size(output_path)
            
            return combined_sample
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def _create_parking_overview(self, df, grid_size=0.005):
        """Create grid-based overview"""
        try:
            minx, miny, maxx, maxy = df.total_bounds
            import numpy as np
            x_coords = np.arange(minx, maxx, grid_size)
            y_coords = np.arange(miny, maxy, grid_size)
            
            polygons = []
            for x in x_coords:
                for y in y_coords:
                    polygons.append(box(x, y, x + grid_size, y + grid_size))
            
            grid = gpd.GeoDataFrame({'geometry': polygons}, crs=df.crs)
            joined = gpd.sjoin(grid, df, how='left', predicate='intersects')
            aggregated = joined.groupby(joined.index).agg({
                'total_spaces': 'sum',
                'paid_spaces': 'sum',
                'has_paid_parking': 'any'
            }).reset_index()
            
            overview = grid.merge(aggregated, left_index=True, right_on='index', how='left')
            overview['total_spaces'] = overview['total_spaces'].fillna(0).astype(int)
            overview['paid_spaces'] = overview['paid_spaces'].fillna(0).astype(int)
            overview['has_paid_parking'] = overview['has_paid_parking'].fillna(False)
            overview = overview[overview['total_spaces'] > 0]
            
            return overview
            
        except Exception as e:
            print(f"    Warning: Could not create overview: {e}")
            return None
    
    # Helper methods
    def _parse_time_limit(self, time_str):
        """Parse time limit to minutes"""
        if pd.isna(time_str) or time_str == '':
            return None
        
        time_str = str(time_str).lower().strip()
        
        try:
            if 'hour' in time_str:
                hours = float(re.findall(r'\d+\.?\d*', time_str)[0])
                return int(hours * 60)
            elif 'min' in time_str:
                return int(re.findall(r'\d+', time_str)[0])
            elif time_str.isdigit():
                return int(time_str)
        except:
            pass
        
        return None
    
    def _clean_currency(self, value):
        """Clean currency to float"""
        if pd.isna(value):
            return None
        try:
            cleaned = re.sub(r'[^\d.]', '', str(value))
            return float(cleaned) if cleaned else None
        except:
            return None
    
    def _standardize_time(self, time_str):
        """Standardize to 24-hour format"""
        if pd.isna(time_str):
            return None
        
        time_str = str(time_str).strip().upper()
        
        try:
            if 'AM' in time_str or 'PM' in time_str:
                from datetime import datetime
                for fmt in ['%I:%M %p', '%I%p', '%I:%M%p']:
                    try:
                        parsed = datetime.strptime(time_str, fmt)
                        return parsed.strftime('%H:%M')
                    except:
                        continue
            
            if ':' in time_str:
                return time_str.split()[0]
        except:
            pass
        
        return None
    
    def _categorize_price(self, rate):
        """Categorize price"""
        if pd.isna(rate) or rate == 0:
            return 'FREE'
        elif rate < 2.0:
            return 'LOW'
        elif rate < 4.0:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _clean_capacity(self, value):
        """Extract numeric capacity"""
        if pd.isna(value):
            return None
        try:
            numbers = re.findall(r'\d+', str(value))
            return int(numbers[0]) if numbers else None
        except:
            return None
    
    def _print_file_size(self, filepath):
        """Print file size"""
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        status = "✓" if size_mb < 5 else "⚠️"
        print(f"  {status} File size: {size_mb:.2f} MB")
        if size_mb > 5:
            print(f"    Warning: Large file may affect performance")
    
    def verify_all_datasets(self):
        """Verify all cleaned datasets"""
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        
        files = [
            'street_parking_detailed.geojson',
            'street_parking_overview.geojson',
            'parking_tiers_clean.geojson',
            'garages_clean.geojson',
            'parking_all_points.geojson'
        ]
        
        total_size = 0
        
        for filename in files:
            filepath = os.path.join(self.clean_dir, filename)
            if os.path.exists(filepath):
                try:
                    gdf = gpd.read_file(filepath)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    total_size += size_mb
                    
                    print(f"\n{filename}:")
                    print(f"  Features: {len(gdf)}")
                    print(f"  CRS: {gdf.crs}")
                    print(f"  Geometry: {gdf.geometry.type.unique()[0]}")
                    print(f"  Valid: {gdf.geometry.is_valid.all()}")
                    print(f"  Size: {size_mb:.2f} MB")
                    cols = [c for c in gdf.columns if c != 'geometry'][:5]
                    print(f"  Key columns: {', '.join(cols)}")
                    
                except Exception as e:
                    print(f"\n{filename}: Error - {e}")
            else:
                print(f"\n{filename}: Not found (may have been skipped)")
        
        print(f"\n{'=' * 70}")
        print(f"Total size: {total_size:.2f} MB")
        print(f"{'=' * 70}\n")
    
    def run_all(self):
        """Run complete pipeline"""
        print("\nStarting data cleaning pipeline...\n")
        
        self.clean_blockface_comprehensive()
        self.clean_parking_tiers()
        self.clean_garages()
        self.create_combined_dataset()
        
        self.verify_all_datasets()
        
        print("\n" + "=" * 70)
        print("✓ DATA CLEANING COMPLETE!")
        print("=" * 70)
        print(f"\nCleaned files are in: {self.clean_dir}/")
        print("\nDatasets created:")
        print("  • street_parking_detailed.geojson - Full street parking data")
        print("  • street_parking_overview.geojson - Grid overview for far zoom")
        print("  • garages_clean.geojson - Public garages (if available)")
        print("  • parking_tiers_clean.geojson - Pricing zones (if available)")
        print("  • parking_all_points.geojson - Combined overview points")


def main():
    """Main execution"""
    cleaner = SeattleParkingCleaner(raw_dir='raw_data', clean_dir='assets')
    cleaner.run_all()


if __name__ == "__main__":
    main()