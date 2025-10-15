import pandas as pd
import numpy as np
from collections import defaultdict
import random
import copy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import re
from typing import Optional, List
from pydantic import BaseModel

# Define NoteItem class
class NoteItem(BaseModel):
    note_text: str
    note_id: str
    binary_label: Optional[int] = None
    augmented: bool = False
    mrn: Optional[str] = None
    race: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[float] = None

# EnhancedSimpleAugmenter
class EnhancedSimpleAugmenter:

    def __init__(self, dataset: List[NoteItem], n_aug: int = 3, seed: int = 42, 
                 parallel: bool = True, max_workers: int = 4):
        self.dataset = dataset
        self.n_aug = n_aug
        self.new_dataset = []
        self.parallel = parallel
        self.max_workers = max_workers
        random.seed(seed)
        self.local = threading.local()
        
        self.modifier_patterns = [
            r'\bmild\b', r'\bmoderate\b', r'\bsevere\b',
            r'\bminimal\b', r'\bmarked\b', r'\bpronounced\b',
            r'\blikely\b', r'\bpossible\b', r'\bprobable\b', r'\bsuspected\b',
            r'\bearly\b', r'\badvanced\b', r'\bslight\b', r'\bsignificant\b',
            r'\binitial\b', r'\bprogressive\b', r'\bstable\b', r'\bworsening\b',
            r'\bimproved\b', r'\bdegenerated\b'
        ]

        self.synonym_dict = {
            "intraocular pressure": ["IOP", "eye pressure", "ocular pressure"],
            "optic nerve": ["optic disc", "nerve head", "ONH", "optic papilla"],
            "visual field": ["VF", "field of vision", "perimetry result", "visual field test"],
            "retinal nerve fiber layer": ["RNFL", "nerve fiber layer", "NFL"],
            "fundus examination": ["funduscopy", "retinal exam", "posterior segment exam"],
            "macular edema": ["ME", "macular swelling", "retinal thickening"],
            "best corrected visual acuity": ["BCVA", "corrected vision", "best vision"],
            "optical coherence tomography": ["OCT", "retinal OCT", "ocular imaging"],
            "anterior chamber": ["AC", "anterior segment", "anterior eye chamber"],
            "posterior capsular opacification": ["PCO", "after-cataract", "secondary cataract"],
            "diabetic retinopathy": ["DR", "diabetic eye disease", "diabetes eye changes"],
            "age-related macular degeneration": ["AMD", "ARMD", "macular degeneration"],
            "primary open-angle glaucoma": ["POAG", "open-angle glaucoma", "chronic glaucoma"],
            "central retinal vein occlusion": ["CRVO", "retinal vein occlusion"],
            "corneal thickness": ["pachymetry", "corneal measurement", "corneal pachymetry"],
            "intraocular lens": ["IOL", "lens implant", "artificial lens"],
            "cup-to-disc ratio": ["C/D ratio", "CD ratio", "cupping ratio"],
        }
        
        self.measurement_dict = {
            r'(\d+)\s*mmHg': [r'\1 mm Hg', r'\1mmHg', r'\1 millimeters of mercury'],
            r'(\d+)/(\d+)': [r'\1 out of \2', r'\1/\2', r'\1:\2'],
            r'(\d+\.?\d*)\s*mm': [r'\1 millimeters', r'\1mm', r'\1 mm'],
            r'(\d+\.?\d*)\s*diopters': [r'\1 D', r'\1D', r'\1 diopter'],
            r'(\d+\.?\d*)\s*degrees': [r'\1°', r'\1 deg', r'\1 degrees'],
            r'(\d+)%': [r'\1 percent', r'\1%', r'\1 pct'],
        }
        
        self.abbreviation_dict = {
            r'\bOD\b': ["right eye", "oculus dexter"],
            r'\bOS\b': ["left eye", "oculus sinister"],
            r'\bOU\b': ["both eyes", "oculus uterque"],
            r'\bVA\b': ["visual acuity", "vision"],
            r'\bAOD\b': ["anterior opening distance"],
            r'\bCCT\b': ["central corneal thickness"],
            r'\bPRL\b': ["preferred retinal locus"],
            r'\bCF\b': ["count fingers"],
            r'\bHM\b': ["hand motion"],
            r'\bLP\b': ["light perception"],
            r'\bNLP\b': ["no light perception"],
            r'\bPDR\b': ["proliferative diabetic retinopathy"],
            r'\bNPDR\b': ["non-proliferative diabetic retinopathy"],
            r'\bCRVO\b': ["central retinal vein occlusion"],
            r'\bBRVO\b': ["branch retinal vein occlusion"],
            r'\bDME\b': ["diabetic macular edema"],
            r'\bCME\b': ["cystoid macular edema"],
            r'\bERM\b': ["epiretinal membrane"],
            r'\bPVD\b': ["posterior vitreous detachment"],
            r'\bRRD\b': ["rhegmatogenous retinal detachment"],
        }
        
        self.section_headers = [
            "History of Present Illness", "HPI", 
            "Past Medical History", "PMH",
            "Ocular History", "Past Ocular History",
            "Family History", "Family Ocular History",
            "Visual Acuity", "VA",
            "External Examination", "Ext",
            "Slit Lamp Examination", "SLE",
            "Tonometry", "IOP Measurement",
            "Fundus Examination", "Dilated Fundus Exam", "DFE",
            "Assessment", "Impression", "Diagnosis",
            "Plan", "Treatment Plan", "Recommendations"
        ]
        
        self.neutral_statements = [
            "Patient records reviewed.",
            "Previous visit information noted.",
            "Patient history considered in assessment.",
            "All findings documented in EMR.",
            "Exam performed using standard protocols.",
            "Follow-up schedule discussed with patient.",
            "No adverse reactions to testing procedures.",
            "All questions addressed during visit.",
            "Patient understanding confirmed.",
            "Standard documentation completed."
        ]
        
        self.procedure_descriptions = {
            "OCT scanning": ["optical coherence tomography imaging", "retinal OCT scan", "OCT assessment"],
            "fundus photography": ["retinal photography", "fundus imaging", "retinal documentation"],
            "gonioscopy": ["angle assessment", "angle evaluation", "gonioscopic examination"],
            "perimetry": ["visual field testing", "automated perimetry", "field of vision assessment"],
            "pachymetry": ["corneal thickness measurement", "corneal assessment", "thickness evaluation"],
            "refraction": ["vision correction assessment", "prescription determination", "refractive evaluation"]
        }

    def _get_local_random(self):
        if not hasattr(self.local, 'random'):
            self.local.random = random.Random(random.randint(0, 2**32 - 1))
        return self.local.random

    def _choose_random_transformations(self) -> List[str]:
        rand = self._get_local_random()
        all_transforms = [
            "_shuffle_sections", 
            "_drop_modifiers",
            "_replace_synonyms", 
            "_swap_abbreviations",
            "_vary_measurements",
            "_inject_noise",
            "_add_neutral_statement",
            "_vary_procedure_descriptions"
        ]
        num_transforms = rand.randint(2, 5)
        return rand.sample(all_transforms, num_transforms)

    def _identify_sections(self, text: str) -> List[str]:
        sections = []
        lines = text.split('\n')
        current_section = []
        
        for line in lines:
            is_header = False
            for header in self.section_headers:
                if re.search(rf'\b{re.escape(header)}:?\b', line, flags=re.IGNORECASE):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    is_header = True
                    current_section.append(line)
                    break
            if not is_header and current_section:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        if not sections:
            if '\n\n' in text:
                sections = text.split('\n\n')
            else:
                sections = [s.strip() + '.' for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        return sections

    def _shuffle_sections(self, text: str) -> str:
        sections = self._identify_sections(text)
        if len(sections) <= 1:
            return text
        rand = self._get_local_random()
        
        header_indices = []
        for i, section in enumerate(sections):
            for header in self.section_headers:
                if re.search(rf'\b{re.escape(header)}:?\b', section.split('\n')[0], flags=re.IGNORECASE):
                    header_indices.append(i)
                    break
        
        if header_indices:
            grouped_sections = []
            last_idx = 0
            for idx in header_indices:
                if idx > last_idx:
                    grouped_sections.append(sections[last_idx:idx])
                if idx < len(sections) - 1 and idx + 1 not in header_indices:
                    grouped_sections.append([sections[idx], sections[idx+1]])
                    last_idx = idx + 2
                else:
                    grouped_sections.append([sections[idx]])
                    last_idx = idx + 1
            if last_idx < len(sections):
                grouped_sections.append(sections[last_idx:])
            
            rand.shuffle(grouped_sections)
            shuffled_sections = []
            for group in grouped_sections:
                shuffled_sections.extend(group)
            return '\n\n'.join(shuffled_sections)
        else:
            rand.shuffle(sections)
            return '\n\n'.join(sections)

    def _drop_modifiers(self, text: str) -> str:
        for pattern in self.modifier_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return re.sub(r'\s{2,}', ' ', text)

    def _replace_synonyms(self, text: str) -> str:
        rand = self._get_local_random()
        result = text
        for phrase, synonyms in self.synonym_dict.items():
            if re.search(rf'\b{re.escape(phrase)}\b', result, flags=re.IGNORECASE):
                replacement = rand.choice(synonyms)
                result = re.sub(rf'\b{re.escape(phrase)}\b', replacement, result, flags=re.IGNORECASE)
        return result

    def _swap_abbreviations(self, text: str) -> str:
        rand = self._get_local_random()
        result = text
        abbreviation_patterns = list(self.abbreviation_dict.keys())
        rand.shuffle(abbreviation_patterns)
        patterns_to_use = abbreviation_patterns[:max(1, len(abbreviation_patterns) // 3)]
        for pattern in patterns_to_use:
            if re.search(pattern, result, flags=re.IGNORECASE):
                replacement = rand.choice(self.abbreviation_dict[pattern])
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def _vary_measurements(self, text: str) -> str:
        rand = self._get_local_random()
        result = text
        for pattern, alternatives in self.measurement_dict.items():
            matches = list(re.finditer(pattern, result))
            if matches:
                indices_to_modify = rand.sample(range(len(matches)), max(1, len(matches) // 2))
                for idx in sorted(indices_to_modify, reverse=True):
                    match = matches[idx]
                    span = match.span()
                    replacement_template = rand.choice(alternatives)
                    groups = match.groups()
                    replacement = replacement_template
                    for i, group in enumerate(groups):
                        replacement = replacement.replace(f'\\{i+1}', group)
                    result = result[:span[0]] + replacement + result[span[1]:]
        return result

    def _inject_noise(self, text: str) -> str:
        rand = self._get_local_random()
        num_ops = rand.randint(1, 3)
        noise_ops = [
            lambda t: re.sub(r'(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.)\s', '.  ', t),
            lambda t: re.sub(r',(?!\s)', ', ', t),
            lambda t: re.sub(r'(\w)-(\w)', r'\1\2', t) if rand.random() < 0.5 else re.sub(r'(\w)(\w)', r'\1-\2', t, count=rand.randint(1, 3)),
            lambda t: re.sub(r';\s*', '; ', t),
            lambda t: re.sub(r'\b(and|with|without|the|to|in|of|for|on|at)\b', 
                             lambda m: m.group(0).capitalize() if rand.random() < 0.2 else m.group(0), 
                             t, flags=re.IGNORECASE)
        ]
        selected_ops = rand.sample(noise_ops, min(num_ops, len(noise_ops)))
        result = text
        for op in selected_ops:
            result = op(result)
        return result

    def _add_neutral_statement(self, text: str) -> str:
        rand = self._get_local_random()
        statement = rand.choice(self.neutral_statements)
        sections = self._identify_sections(text)
        if len(sections) <= 1:
            return text + "\n" + statement
        insert_idx = rand.randint(1, len(sections) - 1)
        result = "\n\n".join(sections[:insert_idx])
        result += "\n\n" + statement + "\n\n"
        result += "\n\n".join(sections[insert_idx:])
        return result

    def _vary_procedure_descriptions(self, text: str) -> str:
        rand = self._get_local_random()
        result = text
        for procedure, alternatives in self.procedure_descriptions.items():
            if re.search(rf'\b{re.escape(procedure)}\b', result, flags=re.IGNORECASE):
                replacement = rand.choice(alternatives)
                matches = list(re.finditer(rf'\b{re.escape(procedure)}\b', result, flags=re.IGNORECASE))
                if matches:
                    num_to_replace = max(1, int(len(matches) * 0.7))
                    indices_to_modify = rand.sample(range(len(matches)), num_to_replace)
                    for idx in sorted(indices_to_modify, reverse=True):
                        match = matches[idx]
                        span = match.span()
                        result = result[:span[0]] + replacement + result[span[1]:]
        return result

    def _augment_single_note(self, note: NoteItem) -> List[NoteItem]:
        results = [note]
        for i in range(self.n_aug):
            new_note = copy.deepcopy(note)
            augmented_text = new_note.note_text
            transformations = self._choose_random_transformations()
            for transform_name in transformations:
                transform_func = getattr(self, transform_name)
                augmented_text = transform_func(augmented_text)
            new_note.note_text = augmented_text.strip()
            new_note.augmented = True
            if hasattr(new_note, 'note_id') and new_note.note_id:
                new_note.note_id = f"{new_note.note_id}_aug{i+1}"
            results.append(new_note)
        return results

    def augment(self) -> List[NoteItem]:
        if self.parallel and len(self.dataset) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                augmented_results = list(executor.map(self._augment_single_note, self.dataset))
            for result_set in augmented_results:
                self.new_dataset.extend(result_set)
        else:
            for note in tqdm(self.dataset, desc="Augmenting notes"):
                self.new_dataset.extend(self._augment_single_note(note))
        return self.new_dataset

# Helper functions
def prepare_labeled_dataset(df):
    note_items = []
    for _, row in df.iterrows():
        note_item = NoteItem(
            note_text=str(row['input_txt']),
            note_id=str(row['noteid']),
            binary_label=int(row['output']),
            mrn=str(row['MRN']),
            race=str(row['race']) if pd.notna(row['race']) else None,
            gender=str(row['gender']) if pd.notna(row['gender']) else None,
            age=float(row['age']) if pd.notna(row['age']) else None
        )
        note_items.append(note_item)
    return note_items

def prepare_unlabeled_dataset(df):
    note_items = []
    for _, row in df.iterrows():
        note_item = NoteItem(
            note_text=str(row['note_txt']),
            note_id=str(row['NOTE_ID']),
            binary_label=None,
            mrn=str(row['mrn']),
            race=str(row['race']) if pd.notna(row['race']) else None,
            gender=str(row['gender']) if pd.notna(row['gender']) else None,
            age=float(row['age']) if pd.notna(row['age']) else None
        )
        note_items.append(note_item)
    return note_items

def note_items_to_labeled_df(note_items):
    data = []
    for item in note_items:
        data.append({
            'MRN': item.mrn,
            'noteid': item.note_id,
            'input_txt': item.note_text,
            'output': item.binary_label,
            'race': item.race,
            'gender': item.gender,
            'age': item.age
        })
    return pd.DataFrame(data)

def note_items_to_unlabeled_df(note_items):
    data = []
    for item in note_items:
        data.append({
            'mrn': item.mrn,
            'NOTE_ID': item.note_id,
            'note_txt': item.note_text,
            'race': item.race,
            'gender': item.gender,
            'age': item.age
        })
    return pd.DataFrame(data)

def get_balanced_augmentation_plan(df, target_size_per_group=400, min_aug=3, max_aug=50):
    """
    Creates augmentation plan to balance the dataset across race-output combinations.
    
    Args:
        df: DataFrame with 'race' and 'output' columns
        target_size_per_group: Target number of samples per race-output combination
        min_aug: Minimum number of augmentations even for larger groups
        max_aug: Maximum number of augmentations to prevent extreme cases
    
    Returns:
        Dictionary with augmentation plan
    """
    print("=== Balanced Augmentation Plan ===")
    
    # Ensure data types are correct
    df_copy = df.copy()
    df_copy['race'] = df_copy['race'].astype(int)
    df_copy['output'] = df_copy['output'].astype(int)
    
    # Calculate group counts
    group_counts = df_copy.groupby(['race', 'output']).size().reset_index(name='count')
    print("Current group counts:")
    print(group_counts.pivot(index='race', columns='output', values='count').fillna(0))
    
    aug_plan = {}
    print(f"\nAugmentation plan (target: {target_size_per_group} per group, max_aug: {max_aug}):")
    
    for _, row in group_counts.iterrows():
        race, output, current_count = int(row['race']), int(row['output']), row['count']
        
        # Calculate needed samples to reach target
        needed = target_size_per_group - current_count
        
        if needed > 0:
            # Calculate n_aug to reach target
            n_aug = max(min_aug, int(np.ceil(needed / current_count)))
            # Cap the augmentation to prevent extreme cases
            n_aug = min(n_aug, max_aug)
        else:
            # Even for groups that are already large, do minimum augmentation
            n_aug = min_aug
        
        aug_plan[(output, race)] = n_aug
        expected_total = current_count * (1 + n_aug)
        
        print(f"   Race {race}, Output {output}: {current_count} → {expected_total} (n_aug={n_aug})")
    
    return aug_plan

def analyze_augmentation_impact(df, aug_plan):
    """
    Analyze the expected impact of the augmentation plan
    """
    print("\n=== Augmentation Impact Analysis ===")
    
    df_copy = df.copy()
    df_copy['race'] = df_copy['race'].astype(int)
    df_copy['output'] = df_copy['output'].astype(int)
    
    group_counts = df_copy.groupby(['race', 'output']).size().reset_index(name='count')
    
    print("Before augmentation:")
    before = group_counts.pivot(index='race', columns='output', values='count').fillna(0)
    print(before)
    
    print("\nAfter augmentation (projected):")
    after_data = []
    for _, row in group_counts.iterrows():
        race, output, current_count = int(row['race']), int(row['output']), row['count']
        n_aug = aug_plan.get((output, race), 3)
        new_count = current_count * (1 + n_aug)
        after_data.append({'race': race, 'output': output, 'count': new_count})
    
    after_df = pd.DataFrame(after_data)
    after = after_df.pivot(index='race', columns='output', values='count').fillna(0)
    print(after)
    
    print("\nBalance improvement:")
    print("Before - CV (coefficient of variation) per race:")
    for race in before.index:
        cv_before = before.loc[race].std() / before.loc[race].mean()
        print(f"   Race {race}: {cv_before:.3f}")
    
    print("After - CV (coefficient of variation) per race:")
    for race in after.index:
        cv_after = after.loc[race].std() / after.loc[race].mean()
        print(f"   Race {race}: {cv_after:.3f}")

def augment_training_data_only(
    train_set, unlabeled_train_bal,
    target_size_per_group=400, min_aug=3, max_aug=50,
    seed=42, parallel=True, max_workers=4
):
    """
    Augment ONLY training (labeled and unlabeled) data.
    Validation and test sets are NOT augmented.
    
    Args:
        train_set: Training set DataFrame
        unlabeled_train_bal: Unlabeled training DataFrame
        target_size_per_group: Target samples per race-output group
        min_aug: Minimum augmentations
        max_aug: Maximum augmentations
        seed: Random seed
        parallel: Use parallel processing
        max_workers: Number of parallel workers
    
    Returns:
        augmented_labeled_df, augmented_unlabeled_df
    """
    random.seed(seed)
    
    print("="*70)
    print("AUGMENTATION: TRAINING SETS ONLY")
    print("(Validation and test sets will NOT be augmented)")
    print("="*70)
    
    # Get augmentation plan for training set
    aug_plan = get_balanced_augmentation_plan(
        train_set, target_size_per_group, min_aug, max_aug
    )
    
    # Analyze the impact
    analyze_augmentation_impact(train_set, aug_plan)
    
    # Prepare datasets
    print("\nPreparing datasets...")
    labeled_note_items = prepare_labeled_dataset(train_set)
    unlabeled_note_items = prepare_unlabeled_dataset(unlabeled_train_bal)
    
    # Group labeled items by (output, race)
    labeled_grouped = defaultdict(list)
    for item in labeled_note_items:
        key = (int(item.binary_label), int(item.race))
        labeled_grouped[key].append(item)
    
    # Apply augmentation to training set
    print("\nApplying augmentation to labeled training set...")
    augmented_labeled_notes = []
    
    for key, items in labeled_grouped.items():
        n_aug = aug_plan.get(key, min_aug)
        print(f"   Processing {key}: {len(items)} items, n_aug={n_aug}")
        
        if n_aug > 0:
            augmenter = EnhancedSimpleAugmenter(
                dataset=items, n_aug=n_aug, seed=seed, 
                parallel=parallel, max_workers=max_workers
            )
            augmented_items = augmenter.augment()
            augmented_labeled_notes.extend(augmented_items)
            print(f"     → Generated {len(augmented_items)} total items")
        else:
            augmented_labeled_notes.extend(items)
            print(f"     → Added {len(items)} original items")
    
    # Convert back to DataFrame
    augmented_labeled_df = note_items_to_labeled_df(augmented_labeled_notes)
    
    # Show final distribution
    print("\n=== FINAL LABELED TRAINING SET DISTRIBUTION ===")
    final_crosstab = pd.crosstab(augmented_labeled_df['race'], augmented_labeled_df['output'])
    print(final_crosstab)
    
    # Process unlabeled data
    print("\nApplying augmentation to unlabeled training set...")
    
    def age_group(age):
        return int(age // 10 * 10) if pd.notna(age) else 'unknown'

    unlabeled_grouped = defaultdict(list)
    for item in unlabeled_note_items:
        key = (str(item.race), str(item.gender), age_group(item.age))
        unlabeled_grouped[key].append(item)

    augmented_unlabeled_notes = []
    for key, items in unlabeled_grouped.items():
        print(f"   Processing unlabeled group {key}: {len(items)} items")
        augmenter = EnhancedSimpleAugmenter(
            dataset=items, n_aug=min_aug, seed=seed, 
            parallel=parallel, max_workers=max_workers
        )
        augmented_items = augmenter.augment()
        augmented_unlabeled_notes.extend(augmented_items)
        print(f"     → Generated {len(augmented_items)} total items")
    
    augmented_unlabeled_df = note_items_to_unlabeled_df(augmented_unlabeled_notes)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Labeled training: {len(train_set)} → {len(augmented_labeled_df)} notes")
    print(f"Unlabeled training: {len(unlabeled_train_bal)} → {len(augmented_unlabeled_df)} notes")
    
    return augmented_labeled_df, augmented_unlabeled_df

# Example usage:
if __name__ == "__main__":
    print("Augmenting ONLY training data (labeled + unlabeled)...\n")
    
    augmented_labeled_df, augmented_unlabeled_df = augment_training_data_only(
        train_set, 
        unlabeled_train_bal,
        target_size_per_group=400,
        min_aug=3,
        max_aug=50,
        seed=42, 
        parallel=True, 
        max_workers=4
    )
    
    print("\n=== TRAINING SET FINAL DISTRIBUTION ===")
    print("Labeled training set:")
    train_crosstab = pd.crosstab(augmented_labeled_df['race'], augmented_labeled_df['output'])
    print(train_crosstab)
    
    print("\n" + "="*70)
    print("✅ Augmentation complete!")
    print(f"✅ Labeled training: {len(augmented_labeled_df)} notes")
    print(f"✅ Unlabeled training: {len(augmented_unlabeled_df)} notes")
    print("⚠️  Validation and test sets were NOT augmented")
    print("="*70)