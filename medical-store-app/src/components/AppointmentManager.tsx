import React, { useState } from 'react';
import { Plus, Check, X, Clock } from 'lucide-react';
import { Appointment } from '../types';
import { getAppointments, saveAppointment, updateAppointment } from '../storage';
import '../styles/AppointmentManager.css';

export default function AppointmentManager() {
  const [appointments, setAppointments] = useState<Appointment[]>(getAppointments());
  const [showForm, setShowForm] = useState(false);
  const [filterStatus, setFilterStatus] = useState<'all' | 'scheduled' | 'completed' | 'cancelled'>('all');
  const [formData, setFormData] = useState<Partial<Appointment>>({
    customerName: '',
    customerPhone: '',
    date: '',
    time: '',
    doctorName: 'Dr. Ayurveda',
    notes: '',
    status: 'scheduled',
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.customerName || !formData.customerPhone || !formData.date || !formData.time) {
      alert('Please fill all required fields');
      return;
    }

    const newAppointment: Appointment = {
      id: `APT-${Date.now()}`,
      customerName: formData.customerName,
      customerPhone: formData.customerPhone,
      date: formData.date,
      time: formData.time,
      doctorName: formData.doctorName || 'Dr. Ayurveda',
      notes: formData.notes,
      status: 'scheduled',
    };

    saveAppointment(newAppointment);
    setAppointments(getAppointments());
    resetForm();
  };

  const resetForm = () => {
    setFormData({
      customerName: '',
      customerPhone: '',
      date: '',
      time: '',
      doctorName: 'Dr. Ayurveda',
      notes: '',
      status: 'scheduled',
    });
    setShowForm(false);
  };

  const handleStatusUpdate = (id: string, status: 'scheduled' | 'completed' | 'cancelled') => {
    updateAppointment(id, status);
    setAppointments(getAppointments());
  };

  const filteredAppointments = filterStatus === 'all'
    ? appointments
    : appointments.filter(a => a.status === filterStatus);

  const stats = {
    total: appointments.length,
    scheduled: appointments.filter(a => a.status === 'scheduled').length,
    completed: appointments.filter(a => a.status === 'completed').length,
    cancelled: appointments.filter(a => a.status === 'cancelled').length,
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled':
        return 'status-scheduled';
      case 'completed':
        return 'status-completed';
      case 'cancelled':
        return 'status-cancelled';
      default:
        return '';
    }
  };

  return (
    <div className="appointment-manager">
      <div className="appointment-header">
        <h2 className="card-title">Clinic Appointments</h2>
        <button className="btn-primary button" onClick={() => setShowForm(true)}>
          <Plus size={18} /> Schedule Appointment
        </button>
      </div>

      {/* Stats */}
      <div className="grid-4">
        <div className="stat-card">
          <div className="stat-label">Total Appointments</div>
          <div className="stat-value">{stats.total}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Scheduled</div>
          <div className="stat-value" style={{ color: '#667eea' }}>{stats.scheduled}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Completed</div>
          <div className="stat-value" style={{ color: '#51cf66' }}>{stats.completed}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Cancelled</div>
          <div className="stat-value" style={{ color: '#ff6b6b' }}>{stats.cancelled}</div>
        </div>
      </div>

      {/* Form Modal */}
      {showForm && (
        <div className="modal-overlay" onClick={() => resetForm()}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3 className="modal-title">Schedule Appointment</h3>

            <form onSubmit={handleSubmit}>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Customer Name *</label>
                  <input
                    type="text"
                    name="customerName"
                    className="form-input"
                    value={formData.customerName || ''}
                    onChange={handleInputChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Phone *</label>
                  <input
                    type="tel"
                    name="customerPhone"
                    className="form-input"
                    value={formData.customerPhone || ''}
                    onChange={handleInputChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Date *</label>
                  <input
                    type="date"
                    name="date"
                    className="form-input"
                    value={formData.date || ''}
                    onChange={handleInputChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Time *</label>
                  <input
                    type="time"
                    name="time"
                    className="form-input"
                    value={formData.time || ''}
                    onChange={handleInputChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Doctor</label>
                  <select
                    name="doctorName"
                    className="form-select"
                    value={formData.doctorName || 'Dr. Ayurveda'}
                    onChange={handleInputChange}
                  >
                    <option value="Dr. Ayurveda">Dr. Ayurveda</option>
                    <option value="Dr. Sharma">Dr. Sharma</option>
                    <option value="Dr. Patel">Dr. Patel</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Notes</label>
                <textarea
                  name="notes"
                  className="form-textarea"
                  value={formData.notes || ''}
                  onChange={handleInputChange}
                  rows={3}
                  placeholder="Add appointment notes..."
                />
              </div>

              <div className="modal-footer">
                <button type="button" className="btn-secondary button" onClick={resetForm}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary button">
                  Schedule
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Filter */}
      <div className="card filter-section">
        <div className="filter-buttons">
          {['all', 'scheduled', 'completed', 'cancelled'].map(status => (
            <button
              key={status}
              className={`filter-btn ${filterStatus === status ? 'active' : ''}`}
              onClick={() => setFilterStatus(status as any)}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Appointments List */}
      <div className="card">
        {filteredAppointments.length === 0 ? (
          <p className="empty-state">No appointments found</p>
        ) : (
          <div className="appointments-list">
            {filteredAppointments.map(appointment => (
              <div key={appointment.id} className={`appointment-item ${getStatusColor(appointment.status)}`}>
                <div className="appointment-main">
                  <div className="appointment-info">
                    <h3>{appointment.customerName}</h3>
                    <div className="appointment-details">
                      <span className="detail">
                        <Clock size={16} />
                        {appointment.date} at {appointment.time}
                      </span>
                      <span className="detail">Dr. {appointment.doctorName}</span>
                    </div>
                    {appointment.notes && (
                      <p className="appointment-notes">{appointment.notes}</p>
                    )}
                  </div>

                  <div className="appointment-status">
                    <span className={`badge ${appointment.status}`}>
                      {appointment.status.charAt(0).toUpperCase() + appointment.status.slice(1)}
                    </span>
                  </div>
                </div>

                {appointment.status === 'scheduled' && (
                  <div className="appointment-actions">
                    <button
                      className="btn-success button"
                      onClick={() => handleStatusUpdate(appointment.id, 'completed')}
                      title="Mark as completed"
                    >
                      <Check size={16} /> Complete
                    </button>
                    <button
                      className="btn-danger button"
                      onClick={() => handleStatusUpdate(appointment.id, 'cancelled')}
                      title="Cancel appointment"
                    >
                      <X size={16} /> Cancel
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
