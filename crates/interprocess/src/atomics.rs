//! Cast shared layout fields to `std::sync::atomic` views for compare-exchange (Cloudtoid wire layout).
//!
//! [`QueueHeader`] and [`MessageHeader`] use `#[repr(C)]`; `i64` / `i32` fields used with atomics must
//! remain at the same offsets as the managed implementation.

use std::sync::atomic::{AtomicI32, AtomicI64};

use crate::layout::{MessageHeader, QueueHeader};

/// Returns an atomic view of [`QueueHeader::write_offset`] for `compare_exchange`.
///
/// # Safety
///
/// `header` must point to a valid, aligned [`QueueHeader`] in the mapped region for `'a`.
pub(crate) unsafe fn queue_header_write_offset<'a>(header: *mut QueueHeader) -> &'a AtomicI64 {
    &*(&(*header).write_offset as *const i64 as *const AtomicI64)
}

/// Returns an atomic view of [`QueueHeader::read_lock_timestamp`] for `compare_exchange` / `store`.
///
/// # Safety
///
/// `header` must point to a valid, aligned [`QueueHeader`] in the mapped region for `'a`.
pub(crate) unsafe fn queue_header_read_lock_timestamp<'a>(
    header: *mut QueueHeader,
) -> &'a AtomicI64 {
    &*(&(*header).read_lock_timestamp as *const i64 as *const AtomicI64)
}

/// Returns an atomic view of [`QueueHeader::read_offset`] for `store`.
///
/// # Safety
///
/// `header` must point to a valid, aligned [`QueueHeader`] in the mapped region for `'a`.
pub(crate) unsafe fn queue_header_read_offset<'a>(header: *mut QueueHeader) -> &'a AtomicI64 {
    &*(&(*header).read_offset as *const i64 as *const AtomicI64)
}

/// Returns an atomic view of [`MessageHeader::state`] for `compare_exchange`.
///
/// # Safety
///
/// `msg` must point to a valid, aligned [`MessageHeader`] in the mapped ring for `'a`.
pub(crate) unsafe fn message_header_state<'a>(msg: *const MessageHeader) -> &'a AtomicI32 {
    &*(&(*msg).state as *const i32 as *const AtomicI32)
}
